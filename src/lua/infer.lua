local data_process = require 'data_process'
require 'data_utils'
require 'utils'
require 'torchx'
require 'optim'

-- element-wise relative error
-- we assume num_future = 1
function relative_error(x, x_hat)
    -- x cannot be 0
    local mask = x:ne(0)
    local mask_nElement = x:ne(0):nonzero():nElement()

    -- first fill x with 1 in 0 of mask
    x:maskedFill(1-mask,1)

    local ratio = torch.cdiv(x_hat, x)  -- x_hat/x
    local difference = 1 - ratio
    local re = torch.abs(difference)

    -- apply mask
    re:maskedFill(1-mask,0)

    assert(x:ne(0):nonzero():nElement()/x:dim() == x:nElement())
    assert(mask_nElement==mask:sum())
    return re, mask, mask_nElement  -- mask has 1 where it is valid
end

-- pred: (bsize, num_future, obj_dim)
-- this_future: (bsize, num_future, obj_dim)
-- assume they are normalized
function angle_magnitude(pred, batch, within_batch)
    local this_past, context_past, this_future, context_future, mask = unpack(batch)

    -- first unrelative
    pred = pred:clone():reshape(mp.batch_size, mp.num_future, mp.object_dim)
    pred = data_process.relative_pair(this_past:clone(), pred, true)

    this_future = data_process.relative_pair(this_past:clone(), this_future:clone(), true)

    -- get velocities
    local vx = config_args.si.vx
    local vy = config_args.si.vy
    local vnc = config_args.velocity_normalize_constant

    local pred_vel = (pred[{{},{},{vx,vy}}]:clone()*vnc)  -- (bsize, num_future, 2)
    local gt_vel = (this_future[{{},{},{vx,vy}}]:clone()*vnc)  -- (bsize, num_future, 2)

    -- get magnitudes
    local pred_vel_magnitude = pred_vel:norm(2,3) -- (bsize, num_future, 1)
    local gt_vel_magnitude = gt_vel:norm(2,3) -- (bsize, num_future, 1)
    assert(pred_vel_magnitude:size(2)==1)
    assert(gt_vel_magnitude:size(2)==1)
    local relative_magnitude_error, mask, mask_nElement = relative_error(torch.squeeze(torch.squeeze(gt_vel_magnitude:clone(),2),2), 
                                                    torch.squeeze(torch.squeeze(pred_vel_magnitude:clone(),2),2))  -- (bsize)


    -- get cosine difference
    local numerator = torch.cmul(pred_vel, gt_vel):sum(3) -- (bsize, num_future, 1)
    local denominator = torch.cmul(pred_vel_magnitude,gt_vel_magnitude)  -- (bsize, num_future, 1)
    local cosine_diff = torch.cdiv(numerator,denominator)

    -- local angle = torch.acos(cosine_diff)  -- (bsize, num_future, 1)
    local angle = torch.squeeze(torch.squeeze(cosine_diff,2),2) -- (bsize, num_future, 1)  -- if I do acos then I get nan
    angle:maskedFill(1-mask,0)  -- zero out the ones where velocity was zero

    -- so angle is (bsize, etc, etc)
    -- but you have to be careful

    if within_batch then
        return angle, relative_magnitude_error, mask, mask_nElement
    else
        -- take average
        local avg_angle_error = angle:sum()/mask_nElement
        local avg_relative_magnitude_error = relative_magnitude_error:sum()/mask_nElement
        return avg_angle_error, avg_relative_magnitude_error
    end
end

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
    local hypotheses, si_indices, indices, num_hypotheses, distance_threshold
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
        indices = {1,2,3}  -- only going to use drastic sizes
        hypotheses = generate_onehot_hypotheses(num_hypotheses, indices) -- good, works for batch_size
        distance_threshold = config_args.object_base_size.ball+config_args.velocity_normalize_constant  -- the smallest side of the obstacle. This makes a difference
    elseif property == 'objtype' then
        si_indices = tablex.deepcopy(config_args.si.oid)
        -- si_indices[2] = si_indices[2]-1  -- ignore jenga block
        indices = {1,2}  -- this changes if we are in the invisible world!
        num_hypotheses = si_indices[2]-si_indices[1]+1
        hypotheses = generate_onehot_hypotheses(num_hypotheses, indices) -- good
        distance_threshold = config_args.object_base_size.ball+config_args.velocity_normalize_constant  -- the smallest side of the obstacle. This makes a difference
    elseif property == 'pos_mass_oid_fixedmass' then  -- b2i on context
        -- infer pos, mass in {1, 1e30}, oid in {1,2}
        -- we are doing dras3!
        si_indices = {px=1,py=1,m={1,4},oid={1,2}}
        -- random between 0 and 1 because pos, oid, os are all in that range
        -- {px: rand, py: rand, mass: tensor(4), oid: tensor(3)}
        distance_threshold = config_args.object_base_size.ball+config_args.velocity_normalize_constant
    end

    local accuracy
    if method == 'max_likelihood' then
        accuracy, accuracy_by_speed, accuracy_by_mass = max_likelihood(model, dataloader, params_, hypotheses, si_indices, cf, distance_threshold)
    elseif method == 'max_likelihood_context' then
        accuracy, accuracy_by_speed, accuracy_by_mass = max_likelihood_context(model, dataloader, params_, hypotheses, si_indices, cf, distance_threshold)
    end

    return accuracy, accuracy_by_speed, accuracy_by_mass
end

function property_analysis(model, dataloader, params_, property)
    local si_indices, distance_threshold, property_table
    if property == 'size' then
        si_indices = tablex.deepcopy(config_args.si.os)
        distance_threshold = config_args.object_base_size.ball+config_args.velocity_normalize_constant

        property_table = {}
        property_table[0.5] = {}
        property_table[1] = {}
        property_table[2] = {}
    elseif property == 'objtype' then
        si_indices = tablex.deepcopy(config_args.si.oid)
        distance_threshold = config_args.object_base_size.ball+config_args.velocity_normalize_constant

        property_table = {}
        property_table[1] = {}
        property_table[2] = {} 
    end

    -- you should give the table to the function 
    local avg_properties, num_properties = context_property_analysis(model, dataloader, params_, si_indices, property_table, distance_threshold)

    return avg_properties, num_properties
end

-- copies batch
function apply_hypothesis_onehot(batch, hyp, si_indices, obj_id)
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


function count_correct(batch, ground_truth, best_hypotheses, num_correct, count, cf, distance_threshold, context_mask)
    local hypothesis_length = best_hypotheses:size(2)

    if cf then 
        -- this filter has a 1 if the focus object reverse direction
        local collision_filter_mask = wall_collision_filter(batch, distance_threshold)

        if context_mask then
            -- this context mask has a 1 if the ground truth is from a particular context object
            collision_filter_mask = collision_filter_mask:cmul(context_mask)  -- filter for collisions AND valid context!
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


function find_best_hypotheses(model, params_, batch, hypotheses, si_indices, context_id)
    local best_losses = torch.Tensor(mp.batch_size):fill(math.huge)
    local hypothesis_length = si_indices[2]-si_indices[1]+1
    local best_hypotheses = torch.zeros(mp.batch_size,hypothesis_length)

    for j,h in pairs(hypotheses) do
        local hypothesis_batch = apply_hypothesis_onehot(batch, h, si_indices, context_id)  -- good
        local test_losses, prediction = model:fp_batch(params_, hypothesis_batch, true)  -- good  this should set sim to true!

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
    for i = 1, dataloader.total_batches do
        if mp.debug then xlua.progress(i, dataloader.total_batches) end
        local batch = dataloader:sample_sequential_batch(false)

        local best_hypotheses = find_best_hypotheses(model, params_, batch, hypotheses, si_indices, 0)
        -- now that you have best_hypothesis, compare best_hypotheses with truth
        -- need to construct true hypotheses based on this_past, hypotheses as parameters
        local this_past = batch[1]:clone()
        local ground_truth = torch.squeeze(this_past[{{},{-1},si_indices}])  -- object properties always the same across time
        num_correct, count = count_correct(batch, ground_truth, best_hypotheses, num_correct, count, cf, distance_threshold)

        collectgarbage()
    end
    local accuracy
    if count == 0 then 
        accuracy = 0
    else 
        accuracy =num_correct/count
    end
    print(count..' collisions out of '..dataloader.total_batches*mp.batch_size..' examples')
    return accuracy, accuracy_by_speed, accuracy_by_mass
end

function context_property_analysis(model, dataloader, params_, si_indices, property_table, distance_threshold)

    local num_correct = 0
    local count = 0

    -- here you will keep a bunch of datastructures for all the properties. 
    -- you will add to these as you encounter context objects with those properties

    for i = 1, dataloader.total_batches do
        if mp.debug then xlua.progress(i, dataloader.total_batches) end
        local batch = dataloader:sample_sequential_batch(false)
        local num_context = batch[2]:size(2)

        -- I should do obstacle mask here, and make obstacle mask an argument into context collision filter.
        -- because right now I am assuming that my context object is an obstacle. But actually I can't assume that
        -- since I might be doing inference on object id.

        local valid_contexts = context_collision_filter(batch)

        -- note that here at most one element in valid_contexts per row would be lit up.
        -- so each example in the batch has only one context.
        -- for example: [0, 0, 2, 0, 1] means that examples 1,2,4 have no valid context
        -- and context_id 2 is valid in example 3 and context_id 5 is valid in example 5

        -- good up to here

        for context_id = 1, num_context do
            -- here let's get a onehot mask to see if the context id is in valid_contexts
            local context_mask = valid_contexts:eq(context_id)

            -- to speed up computation
            if context_mask:sum() > 0 then  -- it's okay to use sum because it is a ByteTensor

                -- here get the obstacle mask
                -- TODO! This is important. we need to break it up into si_indices actually
                local obstacle_index, obstacle_mask
                -- if size inference then obstacle mask
                if alleq({si_indices, config_args.si.os}) then
                    obstacle_index = config_args.si.oid[1]+1
                    -- seems to be a problem with resize because resize adds an extra "1". It could be that I'm not looking at the correct part of the memory.
                    -- that is the problem. I didn't make a copy. 
                    obstacle_mask = batch[2][{{},{context_id},{-1},{obstacle_index}}]:reshape(mp.batch_size, 1):byte()  -- (bsize,1)  1 if it is an obstacle 
                    context_mask:cmul(obstacle_mask)
                end

                -- possible todo: if mass inference then don't use obstacles

                local collision_filter_mask = wall_collision_filter(batch, distance_threshold)

                local context_and_wall_mask = torch.cmul(context_mask, collision_filter_mask)

                if context_and_wall_mask:sum() > 0 then
                    local losses, prediction, vel_losses, ang_vel_losses = model:fp_batch(params_, batch, true)  --this should set sim to true!

                    local cd_error, relative_magnitude_error, angle_mask, mask_nElement = angle_magnitude(prediction, batch, true)
                    context_and_wall_mask:cmul(angle_mask)

                    if context_and_wall_mask:sum() > 0 then

                        -- apply context_mask to losses. all are tensors of size (bsize)
                        losses = losses:maskedSelect(context_and_wall_mask)
                        vel_losses = vel_losses:maskedSelect(context_and_wall_mask)
                        ang_vel_losses = ang_vel_losses:maskedSelect(context_and_wall_mask)
                        cd_error = cd_error:maskedSelect(context_and_wall_mask)
                        relative_magnitude_error = relative_magnitude_error:maskedSelect(context_and_wall_mask)

                        -- now all are tensors of size <= bsize

                        -- we know that there is only context, and that particular context has context id
                        -- maskedSelect does things in order
                        local specific_context = extract_context_id_from_batch(batch, context_and_wall_mask, context_id) -- (num_ex_for_context, 1, num_past, obj_dim)

                        local specific_properties = extract_field(specific_context[{{},{},{-1},{}}], si_indices) -- num_valid_contexts. NOTE THAT WE ARE NOT DOING BLOCK TOWER!
                        -- good up to here

                        -- first we figure out which oids and sizes were represented in specific_context
                        -- populate tables

                        for f=1,#specific_properties do
                            -- populate oid
                            table.insert(property_table[specific_properties[f]],
                                    {losses[f], 
                                    vel_losses[f], 
                                    ang_vel_losses[f],
                                    cd_error[f],
                                    relative_magnitude_error[f]})
                        end

                        collectgarbage()
                    end
                end
            end
        end 
    end

    -- now let's do the averaging. we have sizes and oids

    -- transform into tensor (num_samples, 3)
    for t,_ in pairs(property_table) do
        property_table[t] = torch.Tensor(property_table[t])
    end

    -- now let's do averaging
    local avg_properties = {}
    local num_properties = {}

    for t,_ in pairs(property_table) do
        if property_table[t]:nElement() > 0 then
            avg_properties[t] = property_table[t]:mean(1)
            num_properties[t] = property_table[t]:size(1)
        end
    end

    return avg_properties, num_properties
end

-- good
function extract_context_id_from_batch(batch, context_mask, context_id) 
    local this_past, context_past, this_future, context_future, mask = unpack(batch)

    local selected_context = torch.squeeze(context_past[{{},{context_id},{-1}}])  -- the last past timestep  -- good

    -- recall context_mask gives you the example within batch for that particular context_id
    local ex_in_batch_for_context = torch.totable(torch.squeeze(torch.squeeze(context_mask):nonzero(),2))  -- good

    local selected_ex_for_context = {}
    for _,k in pairs(ex_in_batch_for_context) do
        table.insert(selected_ex_for_context,context_past[{{k},{context_id}}]:clone())  -- note that we are cloning here
    end

    selected_ex_for_context = torch.cat(selected_ex_for_context,1)  -- (num_selected_context, 1, num_past, obj_dim)

    return selected_ex_for_context
end

-- good
function extract_field(specific_context, si_indices) 
    local one_hot_field = specific_context[{{},{},{},si_indices}]

    local categories
    if alleq({si_indices, config_args.si.m}) then
        categories = config_args.masses -- TODO! do I want to include all of it or do I want to trim the large mass
    elseif alleq({si_indices, config_args.si.oid}) then
        categories = config_args.oid_ids -- do I want to trim?
    elseif alleq({si_indices, config_args.si.os}) then
        categories = config_args.drastic_object_sizes
    else
        assert(false, 'Unknown property')
    end

    -- now turn one_hot to number
    local fields  = onehot2numall(one_hot_field, categories, mp.cuda)
    fields = torch.squeeze(fields,2) -- only one context
    fields = torch.squeeze(fields,2) -- only take one timestep
    fields = torch.squeeze(fields,2) -- assume the field is only a scalar
    -- now fields is (num_selected_context)

    -- turn it into a table, listed in order for the context
    fields = torch.totable(fields)

    return fields
end


function max_likelihood_context(model, dataloader, params_, hypotheses, si_indices, cf, distance_threshold)
    local num_correct = 0
    local count = 0
    for i = 1, dataloader.total_batches do
        if mp.debug then xlua.progress(i, dataloader.total_batches) end
        local batch = dataloader:sample_sequential_batch(false)
        local num_context = batch[2]:size(2)


        -- I should do obstacle mask here, and make obstacle mask an argument into context collision filter.
        -- because right now I am assuming that my context object is an obstacle. But actually I can't assume that
        -- since I might be doing inference on object id.

        local valid_contexts = context_collision_filter(batch)  -- a (bsize, 1) where elements are the context id

        -- note that here at most one element in valid_contexts per row would be lit up.
        -- so each example in the batch has only one context.
        -- for example: [0, 0, 2, 0, 1] means that examples 1,2,4 have no valid context
        -- and context_id 2 is valid in example 3 and context_id 5 is valid in example 5

        -- good up to here

        for context_id = 1, num_context do
            -- here let's get a onehot mask to see if the context id is in valid_contexts
            local context_mask = valid_contexts:eq(context_id)
            -- to speed up computation
            if context_mask:sum() > 0 then  -- it's okay to use sum because it is a ByteTensor

                -- here get the obstacle mask
                local obstacle_index, obstacle_mask
                -- if size inference then obstacle mask
                if alleq({si_indices, config_args.si.os}) then
                    obstacle_index = config_args.si.oid[1]+1
                    -- seems to be a problem with resize because resize adds an extra "1". It could be that I'm not looking at the correct part of the memory.
                    -- that is the problem. I didn't make a copy. 
                    obstacle_mask = batch[2][{{},{context_id},{-1},{obstacle_index}}]:reshape(mp.batch_size, 1):byte()  -- (bsize,1)  1 if it is an obstacle 
                    context_mask:cmul(obstacle_mask)
                end

                local best_hypotheses = find_best_hypotheses(model, params_, batch, hypotheses, si_indices, context_id)
                -- now that you have best_hypothesis, compare best_hypotheses with truth
                -- need to construct true hypotheses based on this_past, hypotheses as parameters
                local context_past = batch[2]:clone()
                local ground_truth = torch.squeeze(context_past[{{},{context_id},{-1},si_indices}])  -- object properties always the same across time

                -- ground truth: (bsize, hypothesis_length)
                num_correct, count = count_correct(batch, ground_truth, best_hypotheses, num_correct, count, cf, distance_threshold, context_mask)
                collectgarbage()
            end
        end 
    end

    local accuracy
    if count == 0 then 
        accuracy = 0
    else 
        accuracy =num_correct/count
    end
    print(count..' collisions with context out of '..dataloader.total_batches*mp.batch_size..' examples')
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
    local context_basesizes_num = onehot2numall(context_oids, config_args.object_base_size_ids_upper)  -- TODO: incorporate the actual object base size into here!
    
    -- TO SWITCH BETWEEN DRASTIC SIZE AND NORMAL, JUST CHANGE object_sizes in config!
    local context_os_num = onehot2numall(context_os, config_args.drastic_object_sizes)  -- TODO: make sure you distinguish between normal and drastic!
    -- local context_os_num = onehot2numall(context_os, config_args.object_sizes)  -- TODO: make sure you distinguish between normal and drastic!

    -- now squeeze out third and fourth dimensions --> (bsize, num_context)
    context_basesizes_num = torch.squeeze(torch.squeeze(context_basesizes_num,4),3)  -- note that order matters!
    context_os_num = torch.squeeze(torch.squeeze(context_os_num,4),3)  -- note that order matters!

    -- assert(false, 'did you incorporate object base size? and did you take the diagonal into account?')

    local object_sizes = torch.cmul(context_basesizes_num, context_os_num)
    -- good up to here

    -- my only worry is that we won't get enough examples

    return object_sizes
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

    -- manually perform dot product
    local dot = torch.sum(torch.cmul(past_vel, future_vel),2)

    -- you could just only include those for which dot is < 0
    local collision_mask = dot:le(0) -- 1 if collision  -- good

    local pnc = config_args.position_normalize_constant
    local vnc = config_args.velocity_normalize_constant

    -- for wall collision:
    -- get the direction of the velocity at time t. The normal of the wall dotted with that velocity should be positive.
    local px = config_args.si.px
    local py = config_args.si.py
    local past_pos = torch.squeeze(past[{{},{-1},{px,py}}], 2)  -- before collision -- good

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
        -- these are unnormalized!
        -- note that these may go outside of the world boundaries
        -- NOTE! we are using different constants to normalize the velocity and position! We have to unnormalize first
        local fla = past_pos*pnc + past_vel*vnc  -- (bsize, 2)
        local cla = cpast_pos*pnc + cpast_vel*vnc -- (bsize, num_context, 2)

        -- complication! fla and cla may go outside of the world boundaries! Is that a problem?
        -- a context object may also bounce off a wall. Well in that case the context object 
        -- would not hit the focus object. In our scheme, we let the object go out of bounds
        -- because we are not looking at focus objects within the wall boundary, and if a context
        -- object ends up bouncing with the focus object between t and t+1 after it bounces 
        -- off a wall during that same time interval, we'd ignore it because it's cla would be
        -- out of bounds which we won't look at anyways. Note that we are applying the 
        -- wall collision filter only during count_correct. Should we apply it earlier?
        -- AKA, would it affect the code if I did this first? Well, when I apply the wall
        -- collision filter I am just going to ignore that example anyway, so this would
        -- just be extra computation, which is okay.

        -- good up to here

        -- let's compute the euc dist between fla and cla (unnormalized)
        assert(alleq({torch.totable(fla:size()), {mp.batch_size, 2}}))
        assert(alleq({torch.totable(cla:size()), {mp.batch_size, num_context, 2}}))
        local euc_dist_la = compute_euc_dist_o2m(fla, cla)  -- (bsize, num_context)

        -- get the thresholds for each context (bsize, num_context)
        -- WE ARE ASSUMING THAT THE FOCUS OBJECT IS A BALL AND HAS SIZE MULTIPLIER ONE! 
        -- do an assert for this. Need to take object_type and size_multiplier into account.
        -- not that here the distances are unnormalized.
        -- assert for all rows of focus? NOTE THIS! Well, in expand_for_each_object, we've ensured that the focus object will always be a ball
        assert(past[{{},{},{config_args.si.oid[1]}}]:eq(1)) -- make sure it is a ball
        assert(past[{{},{},{config_args.si.os[1]+1}}]:eq(1))  -- make sure size multiplier is 1
        local distance_to_context_edge = config_args.object_base_size.ball  -- note that later we can do something like context_object_sizes for different sized focuses
        local ball_radii = torch.Tensor(mp.batch_size, num_context):fill(distance_to_context_edge) -- this is the base. then we will add in the obstacle sizes

        -- now get the respective obstacle sizes and add that to disance_thresholds (bsize, num_context)
        -- NOTE: this needs to tkae drastic size into account!
        -- we don't need a padding because fla and cla will only exactly equal (size_upper(ball) + size_upper(context)) if the 
        -- collision happens at exactly t+1
        -- the distance threshold is the distance for which the focus object would be exactly
        -- touching the context object (in the case of the ball). Any distance less than this, 
        -- we would potentially collide. I say potentially to cover the case when the ball
        -- is near the flat edge of an obstacle, so it would be within the boundary, but 
        -- it might not collide. Recall that we will only be taking one context, so 
        -- we would throw out examples that have mbiguous cases.
        -- outside this threshold, there is no way we could collide.
        local context_sizes = context_object_sizes(context_past)
        local distance_thresholds = torch.add(ball_radii, context_sizes)

        -- if la is > threshold we take it out. if it is <= threshold, we keep it
        -- 1 if within threshold aka POTENTIAL collision
        local la_within_threshold = (euc_dist_la-distance_thresholds):le(0)

        -- 2. here we will see if the ball is moving toward a context object.
        -- [i,j]=1 if the focus object was moving toward context j of example i
        -- this actually shouldn't matter?
        -- you know actually this might not be the case if the context is moving towards you.
        -- so let's not include it. Is this necessary? Checking fla and cla actually 
        -- is more general than thiss
        -- local moving_toward_context = check_moving_toward_context(past_pos, cpast_pos, past_vel)  -- (bsize, num_context) 

        -- now we apply the following filters. 
        -- a) It must be moving towards context 
        -- b) la must be within threshold
        -- local within_threshold_moving_toward_context = torch.cmul(la_within_threshold, moving_toward_context)  -- (bsize, num_context)

        -- check for each row, and record the indices of that 1. There should only be 1 of thoe indices.
        -- returns a table of size (bsize). Each element of this table is another table with the indices in that row.
        -- if a row doesn't have any indices, then it is an empty table
        -- here we have all the contexts that meet our collision criteria. All we need to do now is to check if there is only one context in an example
        -- local within_threshold_moving_toward_context_indices = torch.find(within_threshold_moving_toward_context,1,2)  -- search over dimension 2

        local la_within_threshold_indices = torch.find(la_within_threshold,1,2)  -- search over dimension 2

        -- here we impose the constraint that only 1 context should be potentially colliding
        local valid_contexts = {}  -- size: batch size
        for ex=1,mp.batch_size do

            local valid_context_ids = la_within_threshold_indices[ex]  -- note that the ordering may not be the same as in future_context_indices.
            if (#valid_context_ids == 1) then
                -- only a valid context if it meets the above criteria
                table.insert(valid_contexts,valid_context_ids[1])
            else 
                table.insert(valid_contexts,0)
            end
        end

        -- now turn valid_contexts into a tensor (bsize, 1).
        valid_contexts = torch.Tensor(valid_contexts):reshape(mp.batch_size,1)  -- this will be your mask

        -- good up to here

        return valid_contexts
    else 
        -- this is not a byte tensor! because it is not supposed to be a binary mask.
        -- this is only for getting the indices of the colliding context object for each example. 
        return collision_mask:float()   -- they are all 0 so we are good
    end
end
