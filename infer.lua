local data_process = require 'data_process'
require 'data_utils'
require 'utils'
require 'torchx'

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
        -- indices = {1,2,3}  -- DRASTIC SIZE INFERENCE ONLY USES TWO VALUES!
        indices = {1,3}  -- DRASTIC SIZE INFERENCE ONLY USES TWO VALUES!
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
        -- HERE NOTE THAT YOU ARE NOT FILTERING FOR OBSTACLE! YOU WILL FILTER LATER!
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
        -- this filter has a 1 if the focus object reverse direction
        local collision_filter_mask = wall_collision_filter(batch, distance_threshold)

        if obstacle_mask then
            -- this obstacle mask has a 1 if the ground truth is from a particular context object
            collision_filter_mask = collision_filter_mask:cmul(obstacle_mask)  -- filter for collisions AND obstacles
        end

        -- after applying both filters, you have the examples in which the focus object
        -- reverses direction and the object whose property you are inferring is an obstacle
        -- you may now proceed.

        local collision_filter_indices = torch.squeeze(collision_filter_mask):nonzero()
        if collision_filter_indices:nElement() > 0 then
            collision_filter_indices = torch.squeeze(collision_filter_indices,2)
            local ground_truth_filtered = ground_truth:clone():index(1,collision_filter_indices)
            local best_hypotheses_filtered = best_hypotheses:clone():index(1,collision_filter_indices)
            local num_pass_through = collision_filter_indices:size(1)
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

        if update_indices:nElement() > 0 then
            update_indices = torch.squeeze(update_indices,2)
            --best_loss should equal test loss at the indices where test loss < best_loss
            best_losses:indexCopy(1,update_indices,test_losses:index(1,update_indices)) -- works
            -- best_hypotheses should equal h at the indices where test loss < best_loss
            best_hypotheses:indexCopy(1,update_indices,torch.repeatTensor(h,update_indices:size(1),1))  -- works
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


        -- I should do obstacle mask here, and make obstacle mask an argument into context collision filter.
        -- because right now I am assuming that my context object is an obstacle. But actually I can't assume that
        -- since I might be doing inference on object id.

        local valid_contexts = context_collision_filter(batch)
        print(valid_contexts)
        -- assert(false)

        for context_id = 1, num_context do
            -- here get the obstacle mask
            local obstacle_index, obstacle_mask
            -- if size inference then obstacle mask
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

    -- you could just only include those for which dot is < 0
    local collision_mask = dot:le(0)
    return collision_mask
end


-- zero out collisions with walls
-- good
function wall_collision_filter(batch, distance_threshold)
    local this_past, context_past, this_future, context_future, mask = unpack(batch)

    -- I could compute manual dot product
    -- this_past: (bsize, numpast, objdim)
    -- this_future: (bsize, numfuture, objdim)
    local past = this_past:clone()
    local future = this_future:clone()
    future = data_process.relative_pair(past, future, true)
    assert(future:size(2)==1)  -- assuming future == 1 at the moment.

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
    local collision_mask = dot:le(0) -- 1 if collision  -- good  (5 x 1)

    -- for wall collision:
    -- get the direction of the velocity at time t. The normal of the wall dotted with that velocity should be positive.
    local px = config_args.si.px
    local py = config_args.si.py
    local future_pos = torch.squeeze(future[{{},{},{px, py}}],2)  -- see where the ball is at the tiem of collision  (bsize, 2)
    local past_pos = torch.squeeze(past[{{},{-1},{px,py}}], 2)  -- before collision

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
    -- close_walls_filter: (bsize, 4). cwf[i,j] = 1 when the focus ball in example i is close to wall j (within the distance threshold)
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

    -- take the inverse. The 1s in the follow mask are the only examples where there exists a possibility of a object collision (NOT WITH A WALL)
    local possible_object_collision = close_to_wall_and_was_going_towards_any_wall_filter:eq(0)  -- (bsize)

    -- do an AND with the collision filter. this will give you the object collision
    local object_collision_mask = torch.cmul(possible_object_collision, collision_mask)

    return object_collision_mask
end

-- unnormalized!
-- return (bsize, num_contex)
local function context_object_sizes(context_past)
    local context = context_past:clone()

    -- the context object is the same across time steps
    local context_oids = context[{{},{},{1},config_args.si.oid}]  -- (bsize, num_context, 1, 3)
    local context_os = context[{{},{},{1},config_args.si.os}]  -- (bsize, num_context, 1, 3)

    -- (bsize, num_context, 1, 1)
    print(config_args.object_base_size_ids_upper)
    print(context_oids)
    local context_oids_num = onehot2numall(context_oids, config_args.object_base_size_ids_upper)  -- TODO: incorporate the actual object base size into here!
    local context_os_num = onehot2numall(context_os, config_args.object_sizes)  -- TODO: make sure you distinguish between normal and drastic!

    -- check that object_base_size_ids works!

    -- now squeeze out third and fourth dimensions --> (bsize, num_context)
    context_oids_num = torch.squeeze(torch.squeeze(context_oids_num,4),3)  -- note that order matters!
    context_os_num = torch.squeeze(torch.squeeze(context_os_num,4),3)  -- note that order matters!

    -- assert(false, 'did you incorporate object base size? and did you take the diagonal into account?')

    local object_sizes = torch.cmul(context_oids_num, context_os_num)
    return object_sizes
end

local function check_moving_toward_context(past_pos, cpast_pos, past_vel)
    -- past_pos: (bsize, 2)
    -- cpast_pos: (bsize, num_context, 2)
    -- past_vel: (bsize, 2)
    local past_pos = past_pos:clone():view(mp.batch_size, 1, 2):expandAs(cpast_pos)
    local past_vel = past_vel:clone():view(mp.batch_size, 1, 2):expandAs(cpast_pos)

    -- first compute the vectors between past_pos and cpast_pos
    local pointing_to_context = cpast_pos - past_pos

    -- next take dot product with past_vel (bsize, num_context)
    local direction_wrt_context = torch.squeeze(torch.sum(torch.cmul(pointing_to_context, past_vel),3),3)

    -- the dot product should be positive if we are moving towards context.
    local is_moving_towards_context = direction_wrt_context:gt(0)  -- bsize, num_context)
    return is_moving_towards_context
end

-- euc dist between "one" and each row of "many"
-- one: (bsize, 2)
-- many: (bsize, num_context, 2)
local function compute_euc_dist_o2m(one, many)
    assert(one:dim()==2 and many:dim()==3)
    local diff = many - one:view(mp.batch_size, 1, 2):expandAs(many)  -- (bsize, num_context, 2)
    local diffsq = torch.pow(diff,2)
    local euc_dists = torch.squeeze(torch.sqrt(diffsq[{{},{},{1}}]+diffsq[{{},{},{2}}]))  -- (bsize, num_context)
    return euc_dists
end


-- returns the id of the context object if the example contains a valid context collision example
-- a context collision example is valid if only 1 context object is within a certain distance threshold of the ball
-- and they collided. This guarantees that that particular context object is the only object that could have possibly collided with 
-- the focus object.
function context_collision_filter(batch)
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
    local collision_mask = dot:le(0) -- 1 if collision  -- good

    -- for wall collision:
    -- get the direction of the velocity at time t. The normal of the wall dotted with that velocity should be positive.
    local px = config_args.si.px
    local py = config_args.si.py
    local past_pos = torch.squeeze(past[{{},{-1},{px,py}}], 2)  -- before collision 

    -- define "focus look ahead" fla = past_pos + past_vel
    -- define "context look ahead" cla = context_past_pos + context_past_vel
    -- now that we know the position and velocity, we will look at the context objects.
    -- for each context object, we first compute the distance between fla and cla. 
    -- if euc_dist(fla, cla) > object_base_size_ids_upper[context]+object_base_size_ids_upper[focus] then that context object is out.
    -- what is left are contexts for which the focus ball POTENTIALLY collides with. We only want one. 
    -- Note that we are under the scenario that we are GIVEN a collision.

    -- only consider looking at the context if there is a collision at all
    if collision_mask:sum() > 0 then
        -- 1. They have to be within (obj_radius + obstacle_diagonal + vnc) of each other

        local cpast = context_past:clone()
        local num_context = context_past:size(2)

        -- context past positions (bsize, num_context, 2)
        local cpast_pos = torch.squeeze(cpast[{{},{},{-1},{px,py}}],3)
        -- context past velocity (bsize, num_context, 2)
        local cpast_vel = torch.squeeze(cpast[{{},{},{-1},{vx,vy}}],3)

        -- Next, let's compute fla and cla: "focus look ahead" and "context look ahead" respectively.
        local fla = past_pos + past_vel  -- (bsize, 2)
        local cla = cpast_pos + cpast_vel -- (bsize, num_context, 2)

        -- let's compute the euc dist between fla and cla (unnormalized)
        assert(alleq({torch.totable(fla:size()), {mp.batch_size, 2}}))
        assert(alleq({torch.totable(cla:size()), {mp.batch_size, num_context, 2}}))
        local euc_dist_la = compute_euc_dist_o2m(fla, cla)*config_args.position_normalize_constant

        -- get the thresholds for each context (bsize, num_context)
        -- WE ARE ASSUMING THAT THE FOCUS OBJECT IS A BALL AND HAS SIZE MULTIPLIER ONE! 
        -- do an assert for this. Need to take object_type and size_multiplier into account.
        -- not that here the distances are unnormalized.
        -- assert for all rows of focus? NOTE THIS! Well, in expand_for_each_object, we've ensured that the focus object will always be a ball
        assert(past[{{},{},{config_args.si.oid[1]}}]:eq(1)) -- make sure it is a ball
        assert(past[{{},{},{config_args.si.os[1]+1}}]:eq(1))  -- make sure size multiplier is 1
        local distance_to_context_edge = config_args.object_base_size.ball  -- note that later we can do something like context_object_sizes for different sized focuses
        local distance_thresholds = torch.Tensor(mp.batch_size, num_context):fill(distance_to_context_edge) -- this is the base. then we will add in the obstacle sizes

        -- now get the respective obstacle sizes and add that to disance_thresholds (bsize, num_context)
        -- NOTE: this needs to tkae drastic size into account!
        -- we don't need a padding because fla and cla will only exactly equal (size_upper(ball) + size_upper(context)) if the 
        -- collision happens at exactly t+1
        local context_sizes = context_object_sizes(context_past)
        distance_thresholds:add(context_sizes)

        -- if la is > threshold we take it out. if it is <= threshold, we keep it
        -- 1 if within threshold aka POTENTIAL collision
        local la_within_threshold = (euc_dist_la-distance_thresholds):le(0)

        -- 2. here we will see if the ball is moving toward a context object.
        -- [i,j]=1 if the focus object was moving toward context j of example i
        local moving_toward_context = check_moving_toward_context(past_pos, cpast_pos, past_vel)  -- (bsize, num_context) 

        -- now we apply the following filters. 
        -- a) It must be moving towards context 
        -- b) la must be within threshold
        local within_threshold_moving_toward_context = torch.cmul(la_within_threshold, moving_toward_context)  -- (bsize, num_context)

        -- check for each row, and record the indices of that 1. There should only be 1 of thoe indices.
        -- returns a table of size (bsize). Each element of this table is another table with the indices in that row.
        -- if a row doesn't have any indices, then it is an empty table
        -- here we have all the contexts that meet our collision criteria. All we need to do now is to check if there is only one context in an example
        local within_threshold_moving_toward_context_indices = torch.find(within_threshold_moving_toward_context,1,2)  -- search over dimension 2

        -- here we impose the constraint that only 1 context should be potentially colliding
        local valid_contexts = {}  -- size: batch size
        for ex=1,mp.batch_size do

            local valid_context_ids = within_threshold_moving_toward_context_indices[ex]  -- note that the ordering may not be the same as in future_context_indices.
            if (#valid_context_ids == 1) then
                -- only a valid context if it meets the above criteria
                table.insert(valid_contexts,valid_context_ids[1])
            else 
                table.insert(valid_contexts,0)
            end
        end

        -- now turn valid_contexts into a tensor (bsize, 1).
        valid_contexts = torch.Tensor(valid_contexts):reshape(mp.batch_size,1)  -- this will be your mask

        return valid_contexts
    else 
        print('no collision')
        -- this is not a byte tensor! because it is not supposed to be a binary mask.
        -- this is only for getting the indices of the colliding context object for each example. 
        return collision_mask:float()   -- they are all 0 so we are good
    end
end
