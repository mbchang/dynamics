local data_process = require 'data_process'
-- local M = require 'branched_variable_obj_model'

-- a table of onehot tensors of size num_hypotheses
function generate_onehot_hypotheses(num_hypotheses)
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
    local hypotheses, si_indices, num_hypotheses
    if property == 'mass' then
        si_indices = tablex.deepcopy(config_args.si.m)
        si_indices[2] = si_indices[2]-1  -- ignore mass 1e30
        num_hypotheses = si_indices[2]-si_indices[1]+1
        hypotheses = generate_onehot_hypotheses(num_hypotheses) -- good
    elseif property == 'size' then 
        si_indices = tablex.deepcopy(config_args.si.os)
        num_hypotheses = si_indices[2]-si_indices[1]+1
        hypotheses = generate_onehot_hypotheses(num_hypotheses) -- good
    elseif property == 'objtype' then
        si_indices = tablex.deepcopy(config_args.si.oid)
        si_indices[2] = si_indices[2]-1  -- ignore jenga block
        num_hypotheses = si_indices[2]-si_indices[1]+1
        hypotheses = generate_onehot_hypotheses(num_hypotheses) -- good
    end

    local accuracy
    if method == 'backprop' then 
        accuracy = backprop2input(model, dataloader, params_, hypotheses, si_indices, cf)
    elseif method == 'max_likelihood' then
        accuracy = max_likelihood(model, dataloader, params_, hypotheses, si_indices, cf)
    elseif method == 'max_likelihood_context' then
        accuracy = max_likelihood_context(model, dataloader, params_, hypotheses, si_indices, cf)
    end
    return accuracy
end

-- copies batch
function apply_hypothesis(batch, hyp, si_indices)
    local this_past, context_past, this_future, context_future, mask = unpack(batch)
    this_past = this_past:clone()
    context_past = context_past:clone()  -- (bsize, num_context, num_past, obj_dim)
    this_future = this_future:clone()
    context_future = context_future:clone()

    -- I should apply the hypothesis to the entire batch?
    -- later I will have to compare to the original to measure accuracy.
    -- well since I'm going to go through all hypotheses anyways it don't have
    -- to worry about things within the batch. But I have to save the original
    -- this_past for ground truth so that I can compare and measure how many 
    -- within the batch, after I applied all my hypotheses, had the best error.
    local num_ex = this_past:size(1)
    local num_past = this_past:size(2)

    this_past[{{},{},si_indices}] = torch.repeatTensor(hyp, num_ex, num_past, 1)
    return {this_past, context_past, this_future, context_future, mask}
end

-- copies batch
function apply_hypothesis_context(batch, hyp, si_indices, context_id)
    local this_past, context_past, this_future, context_future, mask = unpack(batch)
    this_past = this_past:clone()
    context_past = context_past:clone()
    this_future = this_future:clone()
    context_future = context_future:clone()

    local num_ex = context_past:size(1)
    local num_context = context_past:size(2)
    local num_past = context_past:size(3)

    context_past[{{},{context_id},{},si_indices}] = torch.repeatTensor(hyp, num_ex, 1, num_past, 1)
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


function max_likelihood(model, dataloader, params_, hypotheses, si_indices, cf)
    local num_correct = 0
    local count = 0
    for i = 1, dataloader.num_batches do
        if mp.server == 'pc' then xlua.progress(i, dataloader.num_batches) end
        local batch = dataloader:sample_sequential_batch(false)
        local best_losses = torch.Tensor(mp.batch_size):fill(math.huge)
        local best_hypotheses = torch.zeros(mp.batch_size,#hypotheses)
        local hypothesis_length = si_indices[2]-si_indices[1]+1

        for j,h in pairs(hypotheses) do
            local hypothesis_batch = apply_hypothesis(batch, h, si_indices)  -- good
            local test_losses, prediction = model:fp_batch(params_, hypothesis_batch)  -- good

            -- test_loss is a tensor of size bsize
            local update_indices = test_losses:lt(best_losses):nonzero()

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

        -- now that you have best_hypothesis, compare best_hypotheses with truth
        -- need to construct true hypotheses based on this_past, hypotheses as parameters
        local this_past = batch[1]:clone()
        local ground_truth = torch.squeeze(this_past[{{},{-1},si_indices}])  -- object properties always the same across time

        -- print('ground truth size')
        -- print(ground_truth:size())

        if cf then 
            local collision_filter_mask = collision_filter(batch)
            local ground_truth_filtered = ground_truth:maskedSelect(collision_filter_mask:expandAs(ground_truth))  -- this flattens it though!
            -- print('ground truth filtered')
            -- print(ground_truth_filtered:size())
            if ground_truth_filtered:norm() > 0 then
                -- here you can update count
                -- now select only the indices in ground_truth filtered and best_hypotheses to compare
                local best_hypotheses_filtered = best_hypotheses:maskedSelect(collision_filter_mask:expandAs(best_hypotheses))
                -- print('best hypotheses filtered')
                -- print(best_hypotheses_filtered:size())
                -- assert(false)

                local num_pass_through = ground_truth_filtered:size(1)/hypothesis_length
                ground_truth_filtered:resize(num_pass_through, hypothesis_length)
                best_hypotheses_filtered:resize(num_pass_through, hypothesis_length)

                local num_equal = ground_truth_filtered:eq(best_hypotheses_filtered):sum(2):eq(hypothesis_length):sum()  -- (num_pass_through, hypothesis_length)
                num_correct = num_correct + num_equal
                count = count + num_pass_through
            end
        else
            local num_equal = ground_truth:eq(best_hypotheses):sum(2):eq(hypothesis_length):sum()
            num_correct = num_correct + num_equal
            count = count + mp.batch_size
        end
        collectgarbage()
    end
    return num_correct/count
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

            local best_losses = torch.Tensor(mp.batch_size):fill(math.huge)
            local best_hypotheses = torch.zeros(mp.batch_size,#hypotheses)

            -- for this exmaple an obstacle is context_id 2
            for j,h in pairs(hypotheses) do

                local hypothesis_batch = apply_hypothesis_context(batch, h, si_indices, context_id)  -- CHANGED -- good

                local test_losses, prediction = model:fp_batch(params_, hypothesis_batch)  -- good

                -- test_loss is a tensor of size bsize
                local update_indices = test_losses:lt(best_losses):nonzero()

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

            -- now that you have best_hypothesis, compare best_hypotheses with truth
            -- need to construct true hypotheses based on this_past, hypotheses as parameters
            local context_past = batch[2]:clone()
            local ground_truth = torch.squeeze(context_past[{{},{context_id},{-1},si_indices}])  -- object properties always the same across time
            -- ground truth: (bsize, hypothesis_length)

            if cf then 
                local collision_filter_mask = collision_filter(batch)
                local ground_truth_filtered = ground_truth:maskedSelect(collision_filter_mask:expandAs(ground_truth))  -- this flattens it though!
                if ground_truth_filtered:norm() > 0 then
                    -- here you can update count
                    -- now select only the indices in ground_truth filtered and best_hypotheses to compare
                    local best_hypotheses_filtered = best_hypotheses:maskedSelect(collision_filter_mask:expandAs(best_hypotheses))
                    local num_pass_through = ground_truth_filtered:size(1)/hypothesis_length
                    ground_truth_filtered:resize(num_pass_through, hypothesis_length)
                    best_hypotheses_filtered:resize(num_pass_through, hypothesis_length)

                    local num_equal = ground_truth_filtered:eq(best_hypotheses_filtered):sum(2):eq(hypothesis_length):sum()  -- (num_pass_through, hypothesis_length)
                    num_correct = num_correct + num_equal
                    count = count + num_pass_through
                end
            else
                local num_equal = ground_truth:eq(best_hypotheses):sum(2):eq(hypothesis_length):sum()
                num_correct = num_correct + num_equal
                count = count + mp.batch_size
            end
            collectgarbage()


        end 
    end
    return num_correct/count
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