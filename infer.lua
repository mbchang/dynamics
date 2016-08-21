local data_process = require 'data_process'
require 'utils'
-- local M = require 'branched_variable_obj_model'

-- a table of onehot tensors of size num_hypotheses
function generate_onehot_hypotheses(num_hypotheses, indices)
    local hypotheses = {}
    for i=1,#indices do
        local hypothesis = torch.zeros(num_hypotheses)
        hypothesis[{{indices[i]}}]:fill(1)
        table.insert(hypotheses, hypothesis)
    end
    return hypotheses
end

function generate_onehot_hypotheses_orig(num_hypotheses, indices)
    local hypotheses = {}
    for i=1,num_hypotheses do
        local hypothesis = torch.zeros(num_hypotheses)
        hypothesis[{{i}}]:fill(1)
        table.insert(hypotheses, hypothesis)
    end
    return hypotheses
end

function infer_properties(model, dataloader, params_, property, method, cf)
    -- TODO for other properties
    local hypotheses, si_indices, indices, num_hypotheses
    if property == 'mass' then
        si_indices = tablex.deepcopy(config_args.si.m)
        -- si_indices[2] = si_indices[2]-1  -- ignore mass 1e30
        indices = {1,2,3}
        num_hypotheses = si_indices[2]-si_indices[1]+1
        hypotheses = generate_onehot_hypotheses(num_hypotheses,indices) -- good, works for mass
        distance_threshold = config_args.object_base_size.ball+config_args.velocity_normalize_constant  -- because we are basically saying we are drawing a ball-radius-sized buffer around the walls. so we only look at collisions not in that padding.
    elseif property == 'size' then 
        si_indices = tablex.deepcopy(config_args.si.os)
        num_hypotheses = si_indices[2]-si_indices[1]+1
        indices = {1,2,3}
        hypotheses = generate_onehot_hypotheses(num_hypotheses, indices) -- good, works for batch_size
        distance_threshold = config_args.object_base_size.ball+config_args.velocity_normalize_constant  -- the smallest side of the obstacle. This makes a difference
    elseif property == 'objtype' then
        si_indices = tablex.deepcopy(config_args.si.oid)
        -- si_indices[2] = si_indices[2]-1  -- ignore jenga block
        indices = {1,2}  -- this changes if we are in the invisible world!
        num_hypotheses = si_indices[2]-si_indices[1]+1
        hypotheses = generate_onehot_hypotheses(num_hypotheses, indices) -- good
        distance_threshold = config_args.object_base_size.ball+config_args.velocity_normalize_constant  -- the smallest side of the obstacle. This makes a difference
    end

    local accuracy
    if method == 'backprop' then 
        accuracy = backprop2input(model, dataloader, params_, hypotheses, si_indices, cf)
    elseif method == 'max_likelihood' then
        accuracy = max_likelihood(model, dataloader, params_, hypotheses, si_indices, cf, distance_threshold)
    elseif method == 'max_likelihood_context' then
        accuracy = max_likelihood_context(model, dataloader, params_, hypotheses, si_indices, cf, distance_threshold)
    end
    return accuracy
end

-- copies batch
function apply_hypothesis(batch, hyp, si_indices, obj_id)
    local this_past, context_past, this_future, context_future, mask = unpack(batch)
    this_past = this_past:clone()
    context_past = context_past:clone()
    this_future = this_future:clone()
    context_future = context_future:clone()

    local num_ex = context_past:size(1)
    local num_context = context_past:size(2)
    local num_past = context_past:size(3)

    if obj_id == 0 then  -- protocol for this
        -- you should assert that si_indices is for mass.
        assert(alleq({si_indices, config_args.si.m}))
        this_past[{{},{},si_indices}] = torch.repeatTensor(hyp, num_ex, num_past, 1)
    else
        -- here you should see if si_indices are for object type. if your hypothesis is for a ball, then make mass = 1. If 
        local ball_oid_onehot = torch.zeros(#config_args.oid_ids)
        ball_oid_onehot[{{config_args.oid_ids[1]}}]:fill(1)
        if (alleq({si_indices, config_args.si.oid})) and hyp:equal(ball_oid_onehot) then
            local mass_one_hot = torch.zeros(#config_args.masses)
            mass_one_hot[{{1}}]:fill(1) -- select mass=1
            context_past[{{},{obj_id},{},config_args.si.m}] = mass_one_hot:view(1,1,1,#config_args.masses)
                                                                    :expandAs(context_past[{{},{obj_id},{},config_args.si.m}])
        end
        -- now apply the hypothesis as usual
        context_past[{{},{obj_id},{},si_indices}] = torch.repeatTensor(hyp, num_ex, 1, num_past, 1)
    end

    return {this_past, context_past, this_future, context_future, mask}
end


function backprop2input(model, dataloader, params_, si_indices)
    local num_correct = 0
    local count = 0
    local batch_group_size = 1000
    for i = 1, dataloader.num_batches, batch_group_size do
        if mp.server == 'pc' then xlua.progress(i, dataloader.num_batches) end
        
        local batch = dataloader:sample_sequential_batch(false)

        -- return best_hypotheses by backpropagating to the input
        --  initial hypothesis should be 0.5 in all the indices
        local hypothesis_length = si_indices[2]-si_indices[1]+1
        local initial_hypothesis = torch.Tensor(hypothesis_length):fill(0.5)
        local hypothesis_batch = apply_hypothesis(batch, initial_hypothesis, si_indices)  -- good

        local this_past_orig, context_past_orig, this_future_orig, context_future_orig, mask_orig = unpack(hypothesis_batch)

        -- TODO: keep track of whether it is okay to let this_past, context_past, this_future, context_future, mask get mutated!
        -- closure: returns loss, grad_Input
        -- out of scope: batch, si_indices
        function feval_b2i(this_past_hypothesis)
            -- build modified batch
            local updated_hypothesis_batch = {this_past_hypothesis, context_past_orig:clone(), this_future_orig:clone(), context_future_orig:clone(), mask_orig:clone()}

            model.network:clearState() -- zeros out gradInput

            local loss, prediction = model:fp(params_, updated_hypothesis_batch)
            local d_input = model:bp_input(updated_hypothesis_batch,prediction)

            -- 0. Get names
            local d_pairwise = d_input[1]
            local d_identity = d_input[2]

            -- 1. assert that all the d_inputs in pairwise are equal
            local d_focus_in_pairwise = {}
            for i=1,#d_pairwise do
                table.insert(d_focus_in_pairwise, d_pairwise[i][1])
            end
            assert(alleq_tensortable(d_focus_in_pairwise))

            -- 2. Get the gradients that you need to add
            local d_pairwise_focus = d_pairwise[1][1]:clone()  -- pick first one
            local d_identity_focus = d_identity:clone()
            assert(d_pairwise_focus:isSameSizeAs(d_identity_focus))

            -- 3. Add the gradients
            local d_focus = d_pairwise_focus + d_identity_focus

            -- 4. Zero out everything except the property that you are performing inference on
            d_focus:resize(mp.batch_size, mp.num_past, mp.object_dim)
            if si_indices[1] > 1 then
                d_focus[{{},{},{1,si_indices[1]-1}}]:zero()
            end
            if si_indices[2] < mp.object_dim then
                d_focus[{{},{},{1,si_indices[2]+1}}]:zero()
            end
            d_focus:resize(mp.batch_size, mp.num_past*mp.object_dim)

            -- 6. Check that weights have not been changed
                -- check that all the gradParams are 0 in the network
            assert(model.theta.grad_params:norm() == 0)

            collectgarbage()
            return loss, d_focus -- f(x), df/dx
        end

        local num_iters = 10 -- TODO: change this (perhaps you can check for convergence. you just need rough convergence)
        local b2i_optstate = {learningRate = 0.01}  -- TODO tune this 
        local this_past_hypothesis = this_past_orig:clone()

        -- get old model parameters

        for t=1,num_iters do

            local new_this_past_hypothesis, train_loss = optimizer(feval_b2i,
                                this_past_hypothesis, b2i_optstate)  -- next batch

            -- 1. Check that the model parameters have not changed
            assert(model.theta.parameters:equal(old_model_parameters))

            -- 2. Check that the input focus object has changed 
                -- do this outside of feval
                -- very unlikely they are equal, unless gradient was 0
            assert(not(this_past_hypothesis:equal(new_this_past_hypothesis)))

            -- 3. update this_past
            this_past_hypothesis = new_this_past_hypothesis  -- TODO: can just update it above
            collectgarbage()
        end

        -- now you have a this_past as your hypothesis. Select and binarize.
        -- TODO check that this properly gets mutated
        this_past_hypothesis[{{},{},si_indices}] = binarize(this_past_hypothesis[{{},{},si_indices}]) -- NOTE you are assuming the num_future is 1

        -- now that you have best_hypothesis, compare best_hypotheses with truth
        -- need to construct true hypotheses based on this_past, hypotheses as parameters
        local ground_truth = torch.squeeze(this_past_orig[{{},{-1},si_indices}])  -- object properties always the same across time
        local num_equal = ground_truth:eq(this_past_hypothesis[{{},{},si_indices}]):sum(2):eq(hypothesis_length):sum()
        num_correct = num_correct + num_equal
        count = count + mp.batch_size
        collectgarbage()
    end
    return num_correct/count
end

-- selects max over each row in last axis
-- makes the max one and everything else 0
function binarize(tensor)
    local y, i = torch.max(tensor, tensor:dim())
    tensor:zero()
    tensor:indexFill(tensor:dim(), torch.squeeze(i,tensor:dim()), 1)
    return tensor
end


function count_correct(batch, ground_truth, best_hypotheses, hypothesis_length, num_correct, count, cf, distance_threshold, obstacle_mask)
    if cf then 
        local collision_filter_mask = wall_collision_filter(batch, distance_threshold)

        if obstacle_mask then
            collision_filter_mask = collision_filter_mask:cmul(obstacle_mask)  -- filter for collisions AND obstacles
        end

        local ground_truth_filtered = ground_truth:maskedSelect(collision_filter_mask:expandAs(ground_truth))  -- this flattens it though!
        if ground_truth_filtered:norm() > 0 then
            -- here you can update count
            -- now select only the indices in ground_truth filtered and best_hypotheses to compare
            local best_hypotheses_filtered = best_hypotheses:maskedSelect(collision_filter_mask:expandAs(best_hypotheses))

            local num_pass_through = ground_truth_filtered:size(1)/hypothesis_length
            ground_truth_filtered:resize(num_pass_through, hypothesis_length)  -- check this!
            best_hypotheses_filtered:resize(num_pass_through, hypothesis_length)  -- check this!

            local num_equal = ground_truth_filtered:eq(best_hypotheses_filtered):sum(2):eq(hypothesis_length):sum()  -- (num_pass_through, hypothesis_length)
            num_correct = num_correct + num_equal
            count = count + num_pass_through
        end
    else
        local num_equal = ground_truth:eq(best_hypotheses):sum(2):eq(hypothesis_length):sum()
        num_correct = num_correct + num_equal
        count = count + mp.batch_size
    end
    return num_correct, count
end


function find_best_hypotheses(model, params_, batch, hypotheses, hypothesis_length, si_indices, context_id)
    local best_losses = torch.Tensor(mp.batch_size):fill(math.huge)
    local best_hypotheses = torch.zeros(mp.batch_size,hypothesis_length)
    local hypothesis_length = si_indices[2]-si_indices[1]+1

    for j,h in pairs(hypotheses) do
        local hypothesis_batch = apply_hypothesis(batch, h, si_indices, context_id)  -- good
        local test_losses, prediction = model:fp_batch(params_, hypothesis_batch)  -- good

        -- test_loss is a tensor of size bsize
        local update_indices = test_losses:lt(best_losses):nonzero()
        -- print('update_indices', update_indices)

        if update_indices:nElement() > 0 then
            update_indices = torch.squeeze(update_indices,2)
            --best_loss should equal test loss at the indices where test loss < best_loss
            best_losses:indexCopy(1,update_indices,test_losses:index(1,update_indices))

            -- best_hypotheses should equal h at the indices where test loss < best_loss
            best_hypotheses:indexCopy(1,update_indices,torch.repeatTensor(h,update_indices:size(1),1))
        end
        -- check that everything has been updated
        assert(not(best_losses:equal(torch.Tensor(mp.batch_size):fill(math.huge))))
        assert(not(best_hypotheses:equal(torch.zeros(mp.batch_size,hypothesis_length))))
    end
    return best_hypotheses
end

function max_likelihood(model, dataloader, params_, hypotheses, si_indices, cf, distance_threshold)
    local num_correct = 0
    local count = 0
    for i = 1, dataloader.num_batches do
        if mp.server == 'pc' then xlua.progress(i, dataloader.num_batches) end
        local batch = dataloader:sample_sequential_batch(false)

        local hypothesis_length = si_indices[2]-si_indices[1]+1
        local best_hypotheses = find_best_hypotheses(model, params_, batch, hypotheses, hypothesis_length, si_indices, 0)
        -- now that you have best_hypothesis, compare best_hypotheses with truth
        -- need to construct true hypotheses based on this_past, hypotheses as parameters
        local this_past = batch[1]:clone()
        local ground_truth = torch.squeeze(this_past[{{},{-1},si_indices}])  -- object properties always the same across time
        num_correct, count = count_correct(batch, ground_truth, best_hypotheses, hypothesis_length, num_correct, count, cf, distance_threshold)

        collectgarbage()
    end
    local accuracy
    if count == 0 then 
        accuracy = 0
    else 
        accuracy =num_correct/count
    end
    print(count..' collisions out of '..dataloader.num_batches*mp.batch_size..' examples')
    return accuracy
end

function max_likelihood_context(model, dataloader, params_, hypotheses, si_indices, cf)
    local num_correct = 0
    local count = 0
    for i = 1, dataloader.num_batches do
        if mp.server == 'pc' then xlua.progress(i, dataloader.num_batches) end
        local batch = dataloader:sample_sequential_batch(false)
        local hypothesis_length = si_indices[2]-si_indices[1]+1
        local num_context = batch[2]:size(2)

        for context_id = 1, num_context do
            -- here get the obstacle mask
            local obstacle_index, obstacle_mask
            -- if size inference then obstacle mask
            -- if config_args.si.os[1] == si_indices[1] and config_args.si.os[2] == si_indices[2] then
            if alleq({si_indices, config_args.si.os}) then
                obstacle_index = config_args.si.oid[1]+1
                -- seems to be a problem with resize because resize adds an extra "1". It could be that I'm not looking at the correct part of the memory.
                -- that is the problem. I didn't make a copy. 
                obstacle_mask = batch[2][{{},{context_id},{-1},{obstacle_index}}]:reshape(mp.batch_size, 1):byte()  -- (bsize,1)  1 if it is an obstacle 
            end

            local best_hypotheses = find_best_hypotheses(model, params_, batch, hypotheses, hypothesis_length, si_indices, context_id)
            -- now that you have best_hypothesis, compare best_hypotheses with truth
            -- need to construct true hypotheses based on this_past, hypotheses as parameters
            local context_past = batch[2]:clone()
            local ground_truth = torch.squeeze(context_past[{{},{context_id},{-1},si_indices}])  -- object properties always the same across time

            -- ground truth: (bsize, hypothesis_length)
            num_correct, count = count_correct(batch, ground_truth, best_hypotheses, hypothesis_length, num_correct, count, cf, distance_threshold, obstacle_mask)
            collectgarbage()
        end 
    end

    local accuracy
    if count == 0 then 
        accuracy = 0
    else 
        accuracy =num_correct/count
    end
    print(count..' collisions with obstacles out of '..dataloader.num_batches*mp.batch_size..' examples')
    return accuracy
end


-- zero out the examples in which this_past and this_future 
-- are less than the given angle
-- NOTE that when you do forward pass, you'd have to do something different when you average!
-- NOTE that if we do this after we apply neighbor mask, then we could norm that is 0!
-- we have to deal with that. Wait that should be fine, because collision filter just calculates based on batch
-- return input, this_future
function collision_filter(batch)
    local this_past, context_past, this_future, context_future, mask = unpack(batch)

    -- I could compute manual dot product
    -- this_past: (bsize, numpast, objdim)
    -- this_future: (bsize, numfuture, objdim)
    local past = this_past:clone()
    local future = this_future:clone()
    future = data_process.relative_pair(past, future, true)

    local vx = config_args.si.vx
    local vy = config_args.si.vy
    local past_vel = torch.squeeze(past[{{},{-1},{vx, vy}}],2)
    local future_vel = torch.squeeze(future[{{},{},{vx, vy}}],2)

    local past_vel_norm = torch.norm(past_vel,2,2)
    local future_vel_norm = torch.norm(future_vel,2,2)
    local both_norm = torch.cmul(past_vel_norm, future_vel_norm)

    -- manually perform dot product
    local dot = torch.sum(torch.cmul(past_vel, future_vel),2)
    -- local cos_theta = torch.cdiv(dot, both_norm:expandAs(dot)) -- numerical issues here
    -- local theta = torch.acos(cos_theta)

    -- you could just only include those for which dot is < 0
    local collision_mask = dot:le(0)
    return collision_mask
end


-- zero out collisions with walls
function wall_collision_filter(batch, distance_threshold)
    local this_past, context_past, this_future, context_future, mask = unpack(batch)

    -- I could compute manual dot product
    -- this_past: (bsize, numpast, objdim)
    -- this_future: (bsize, numfuture, objdim)
    local past = this_past:clone()
    local future = this_future:clone()
    future = data_process.relative_pair(past, future, true)

    local vx = config_args.si.vx
    local vy = config_args.si.vy
    local past_vel = torch.squeeze(past[{{},{-1},{vx, vy}}],2)  -- (bsize, 2)
    local future_vel = torch.squeeze(future[{{},{},{vx, vy}}],2)

    local past_vel_norm = torch.norm(past_vel,2,2)
    local future_vel_norm = torch.norm(future_vel,2,2)
    local both_norm = torch.cmul(past_vel_norm, future_vel_norm)

    -- manually perform dot product
    local dot = torch.sum(torch.cmul(past_vel, future_vel),2)

    -- you could just only include those for which dot is < 0
    local collision_mask = dot:le(0)

    -- for wall collision:
    -- get the direction of the velocity at time t. The normal of the wall dotted with that velocity should be positive.
    local px = config_args.si.px
    local py = config_args.si.py
    local future_pos = torch.squeeze(future[{{},{},{px, py}}],2)  -- see where the ball is at the tiem of collision  (bsize, 2)
    local past_pos = torch.squeeze(past[{{},{-1},{px,py}}], 2)  -- before collision

    -- print(collision_mask)
    -- print(future_pos*config_args.position_normalize_constant)
    -- print(past_pos*config_args.position_normalize_constant)
    -- print('>>>>>>>>>>>>>>>>>>>>')
    -- assert(false)

    -- now let's check where the wall is
    local leftwall = 0
    local topwall = 0
    local rightwall = 2*config_args.cx
    local bottomwall = 2*config_args.cy
    local walls = (torch.Tensor{leftwall, topwall, rightwall, bottomwall})/config_args.position_normalize_constant  -- size (4)  -- TODO! You have to normalize!

    local leftwall_normal = torch.Tensor({{1,0}})
    local topwall_normal = torch.Tensor({{0,1}})
    local rightwall_normal = torch.Tensor({{-1,0}})
    local bottomwall_normal = torch.Tensor({{0,-1}})
    local wall_normals = torch.cat({leftwall_normal, topwall_normal, rightwall_normal, bottomwall_normal},1)  -- (4,2)

    -- find the nearest wall. this can be found with a simple difference of coordinates
    local future_pos_components = torch.cat({future_pos[{{},{1}}], future_pos[{{},{2}}], future_pos[{{},{1}}], future_pos[{{},{2}}]})  -- (bsize, 4) {x,y,x,y}
    local past_pos_components = torch.cat({past_pos[{{},{1}}], past_pos[{{},{2}}], past_pos[{{},{1}}], past_pos[{{},{2}}]})
    -- local d2leftwall = torch.abs(past_pos[1] - leftwall) -- x
    -- local d2topwall = torch.abs(past_pos[2]- topwall) -- y
    -- local d2rightwall = torch.abs(past_pos[1] - rightwall) -- x
    -- local d2bottomwall = torch.abs(past_pos[2] -bottomwall) --y
    local d2wall = torch.abs(future_pos_components-walls:view(1,4):expandAs(future_pos_components))  -- works  (bisze, 4)
    local d2wallpast = torch.abs(past_pos_components-walls:view(1,4):expandAs(past_pos_components))

    -- filter out the walls that are > distance_threshold away. Perhaps do this in a vector form
    -- select the close wall
    -- ultimately we want to guarantee that we don't collide with a wall
    local close_walls_filter = d2wall:le(distance_threshold/config_args.position_normalize_constant)  -- one ball diameter  (bsize,4)
    close_walls_filter:add(d2wallpast:le(distance_threshold/config_args.position_normalize_constant))  -- filter distance of past as well  -- wait if I add this in I get more examples?
    close_walls_filter:clamp(0,1)
    -- so :cmul does: filtering out the past will have < ones in the close_wall_filters, which means >= ones in the obstacle filter
    -- what that means is that we will kick out the ones that have past AND future by the walls. 
    -- but actually we want the OR. We want the ones whose past OR future are in the walls. Or do we?


    -- dot the wall's normal with your velocity vector
    -- what if two walls are equally close?
    -- past_vel (bsize, 2)
    -- wall_normals (4,2)
    -- result (bsize, 4)
    -- [i,j] means the dot product of the velocity of the ith example with the jth wall
    -- you want the dot product to be negative, because the wall normal points away from the wall
    local dot_with_wall_normal = torch.mm(past_vel, wall_normals:t())  -- (bsize, 4) 
    local towards_wall_filter = dot_with_wall_normal:le(0)

    -- you want to select the examples for which you are close to wall after collision and you were going towards it the previous timestep
    -- it's ok if you don't actually hit the wall in the t+1 timestep. What we are checking is that it should be impossible for you
    -- to hit another ball. So the inverse of this mask filters out all POTENTIAL collisions with walls, leaving true collisions with other objects

    local close_to_wall_and_was_going_towards_some_wall_filter = torch.cmul(close_walls_filter, towards_wall_filter)  -- (bsize, 4)

    -- this figures out which example in the batch has the potential for a wall collision.
    -- do not consider these examples when you do collision filtering, because these rule out the possibility of a ball collision
    local close_to_wall_and_was_going_towards_any_wall_filter = close_to_wall_and_was_going_towards_some_wall_filter:sum(2) -- (bsize)

    -- take the inverse. The 1s in the follow mask are the only examples where there exists a possibility of a ball collision
    local possible_object_collision = close_to_wall_and_was_going_towards_any_wall_filter:eq(0)  -- (bsize)

    -- do an AND with the collision filter. this will give you the object collision
    local object_collision_mask = torch.cmul(possible_object_collision, collision_mask)
    return object_collision_mask
end
