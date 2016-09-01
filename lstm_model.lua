require 'nn'
require 'rnn'
require 'torch'
require 'nngraph'
require 'Base'
require 'IdentityCriterion'
require 'data_utils'
require 'infer'
require 'modules'
local data_process = require 'data_process'

nngraph.setDebug(true)

-- with a bidirectional lstm, no need to put a mask
-- however, you can have variable sequence length now!
function init_network(params)
    -- input: table of length (num_past + num_future - 1) of {(bsize, max_obj*obj_dim), (bsize, max_obj*obj_dim)}  -- just set it to 10
    -- the first element of the inner table is teh data, and the second element of the inner table is the mask

    local rnn_dim = params.rnn_dim
    local obj_dim = params.object_dim
    local max_obj = params.seq_length
    local num_past = params.num_past
    local num_future = params.num_future
    local in_dim = max_obj*obj_dim
    local hid_dim = max_obj*rnn_dim  -- TODO rename rnn_dim to hid_dim
    local out_dim = max_obj*obj_dim  -- note that we will be ignoring the padded areas during backpropagation
    local num_layers = params.layers

    local rnn_core = nn.Sequential()
    if num_layers == 1 then
        rnn_core:add(nn.LSTM(in_dim, out_dim))
    else
        for i = 1, num_layers do -- TODO make sure this is comparable to encoder decoder architecture in terms of layers
            if i == 1 then 
                rnn_core:add(nn.LSTM(in_dim, hid_dim))
                rnn_core:add(nn.ReLU())
            elseif i == num_layers then 
                rnn_core:add(nn.LSTM(hid_dim, out_dim))
            else
                rnn_core:add(nn.LSTM(hid_dim, hid_dim))
                rnn_core:add(nn.ReLU())
            end
        end
    end
    local net = nn.Sequencer(rnn_core)
    return net
end

-- function masker(layertype, activation)
--     local layer_in = nn.Identity()()
--     local mask = nn.Identity()()
--     local lin_out = layertype(layer_in)
--     local lin_out_act = activation(lin_out)
--     local mask_out = nn.CMulTable(){lin_out_act, mask}
--     local layer = nn.gModule({layer_in, mask}, {mask_out, mask})
--     return layer
-- end

--------------------------------------------------------------------------------
--############################################################################--
--------------------------------------------------------------------------------

-- Now create the model class
local model = {}
model.__index = model

function model.create(mp_, preload, model_path)
    local self = {}
    setmetatable(self, model)
    self.mp = mp_

    assert(self.mp.input_dim == self.mp.object_dim * self.mp.num_past)
    assert(self.mp.out_dim == self.mp.object_dim * self.mp.num_future)
    if preload then
        print('Loading saved model.')
        local checkpoint = torch.load(model_path)
        self.network = checkpoint.model.network:clone()
        self.criterion = checkpoint.model.criterion:clone()
        self.identitycriterion = checkpoint.model.identitycriterion:clone()
        if self.mp.cuda then
            self.network:cuda()
            self.criterion:cuda()
            self.identitycriterion:cuda()
        end
    else
        self.criterion = nn.MSECriterion(false)  -- not size averaging!
        self.identitycriterion = nn.IdentityCriterion()
        self.network = init_network(self.mp)
        if self.mp.cuda then
            self.network:cuda()
            self.criterion:cuda()
            self.identitycriterion:cuda()
        end
    end

    self.theta = {}
    self.theta.params, self.theta.grad_params = self.network:getParameters()

    collectgarbage()
    return self
end

function model:pad(tensor, dim, num_to_pad)
    if num_to_pad == 0 then
        return tensor
    else
        local tensor_size = torch.totable(tensor:size())
        assert(dim>=1 and dim<=#tensor_size)

        tensor_size[dim] = num_to_pad
        local padding = torch.zeros(unpack(tensor_size))
        if self.mp.cuda then padding = padding:cuda() end

        local padded = torch.cat({tensor, padding}, dim)
        return padded
    end
end

function model:unpack_batch(batch, sim)
    local this, context, this_future, context_future, mask = unpack(batch)

    -- (bsize, num_obj, timestep, obj_dim)
    local past = torch.cat({unsqueeze(this:clone(),2), context},2)
    local future = torch.cat({unsqueeze(this_future:clone(),2), context_future},2)

    -- (bsize, num_obj, num_past+num_future, obj_dim)
    local trajectories = torch.cat({past, future}, 3)

    -- note that num_future doesn't matter now
    local input = trajectories:clone()[{{},{},{1,-2},{}}]  -- take all but last timestep
    local target = trajectories:clone()[{{},{},{2,-1},{}}]  -- take all but first timestep

    -- YOU HAVE TO PAD!!!!!
    local max_obj = self.mp.seq_length -- TODO: HARDCODED
    local bsize, num_obj, num_past, obj_dim = past:size(1), past:size(2), past:size(3), past:size(4)
    local num_future = future:size(3)

    input = self:pad(input, 2, max_obj-num_obj)  -- (bsize, max_obj, num_past, obj_dim)
    target = self:pad(target, 2, max_obj-num_obj)  -- (bsize, max_obj, num_future, obj_dim)

    -- shuffle the objects
    -- first randperm a binary vector of num_obj 1s and max_obj-num_obj zeros
    -- second randperm the object_id
    -- then put the object_id where the binary vector has 1
    -- IN THAT CASE YOU HAVE TO CHANGE BP!
    local shuffind = torch.randperm(max_obj)
    local zero_ind = {}

    local input_table = {}
    local target_table = {}
    for i = 1, input:size(2) do
        local input_obj = input[{{},{shuffind[i]},{},{}}]
        local target_obj = target[{{},{shuffind[i]},{},{}}]
        if input_obj:norm() == 0 then
            assert(target_obj:norm() == 0)
            table.insert(zero_ind, i)
        end
        table.insert(input_table, input_obj)
        table.insert(target_table, target_obj)
    end
    assert(#zero_ind == max_obj-num_obj)
    input = torch.cat(input_table,2)  -- (bsize, max_obj, (num_past+num_future-1), obj_dim)
    target = torch.cat(target_table,2)

    -- now to break into timestep
    local input_table = {}
    local target_table = {}
    for i = 1, input:size(3) do
        local input_tstep = torch.squeeze(input[{{},{},{i},{}}]):reshape(mp.batch_size, max_obj*obj_dim)
        local target_tstep = torch.squeeze(target[{{},{},{i},{}}]):reshape(mp.batch_size, max_obj*obj_dim)
        table.insert(input_table,input_tstep) -- this works
        table.insert(target_table,target_tstep) -- this works
    end

    -- table of length: (num_past+num_future-1) of (bsize, max_obj*obj_dim)
    return input_table, target_table, num_obj, zero_ind
end


-- Input to fp
-- {
--   1 : DoubleTensor - size: 4x2x9
--   2 : DoubleTensor - size: 4x2x2x9
--   3 : DoubleTensor - size: 4x48x9
--   4 : DoubleTensor - size: 4x2x48x9
--   5 : DoubleTensor - size: 10
-- }
-- good
function model:fp(params_, batch, sim)
    if params_ ~= self.theta.params then self.theta.params:copy(params_) end
    self.theta.grad_params:zero()  -- reset gradient

    local input, target, num_obj, zero_ind = self:unpack_batch(batch, sim)
    local prediction = self.network:forward(input)

    local loss = 0
    local counter = 0
    for i = 1,#prediction do   -- iterate through timesteps
        -- table of length num_obj of {bsize, max_obj*obj_dim}
        local reshaped_prediction = prediction[i]:reshape(mp.batch_size, mp.seq_length, mp.object_dim)
        local reshaped_target = target[i]:reshape(mp.batch_size, mp.seq_length, mp.object_dim)
        -- now iterate through the objects
        for o = 1, mp.seq_length do
            if not(isin(o, zero_ind)) then
                -- split output works
                local prediction_object = reshaped_prediction[{{},{o},{}}]
                local target_object = reshaped_target[{{},{o},{}}]
                local p_pos, p_vel, p_ang, p_ang_vel, p_obj_prop =
                                unpack(split_output(self.mp):forward(prediction_object))
                local gt_pos, gt_vel, gt_ang, gt_ang_vel, gt_obj_prop =
                                unpack(split_output(self.mp):forward(target_object))
                local loss_vel = self.criterion:forward(p_vel, gt_vel)
                local loss_ang_vel = self.criterion:forward(p_ang_vel, gt_ang_vel)
                local obj_loss = loss_vel + loss_ang_vel
                -- obj_loss = obj_loss/(p_vel:nElement()+p_ang_vel:nElement()) -- manually do size average 
                loss = loss + obj_loss
                -- using a counter is equivalent; it's just when you scale
                counter = counter + p_vel:nElement()+p_ang_vel:nElement()
            end
        end
    end
    -- loss = loss/(#prediction*num_obj)  
    loss = loss/counter
    collectgarbage()
    return loss, prediction
end


-- local p_pos, p_vel, p_obj_prop=split_output(params):forward(prediction)
-- local gt_pos, gt_vel, gt_obj_prop=split_output(params):forward(this_future)
-- a lot of instantiations of split_output
function model:bp(batch, prediction, sim)
    self.theta.grad_params:zero() -- the d_parameters
    local input, target, num_obj, zero_ind = self:unpack_batch(batch, sim)

    local splitter = split_output(self.mp)

    local d_pred = {}
    for i = 1, #prediction do

        if i==#prediction then -- only backpropagate from the last timestep

            local reshaped_prediction = prediction[i]:reshape(mp.batch_size, mp.seq_length, mp.object_dim)
            local reshaped_target = target[i]:reshape(mp.batch_size, mp.seq_length, mp.object_dim)

            local d_pred_objs = {}

            for o=1, mp.seq_length do
                if not(isin(o, zero_ind)) then
                    local prediction_object = reshaped_prediction[{{},{o},{}}]
                    local target_object = reshaped_target[{{},{o},{}}]
                    local p_pos, p_vel, p_ang, p_ang_vel, p_obj_prop =
                                    unpack(splitter:forward(prediction_object))
                    local gt_pos, gt_vel, gt_ang, gt_ang_vel, gt_obj_prop =
                                    unpack(split_output(self.mp):forward(target_object))

                    self.identitycriterion:forward(p_pos, gt_pos)
                    local d_pos = self.identitycriterion:backward(p_pos, gt_pos):clone()

                    self.criterion:forward(p_vel, gt_vel)
                    local d_vel = self.criterion:backward(p_vel, gt_vel):clone()
                    d_vel = d_vel/d_vel:nElement()  -- manually do sizeAverage

                    self.identitycriterion:forward(p_ang, gt_ang)
                    local d_ang = self.identitycriterion:backward(p_ang, gt_ang):clone()

                    self.criterion:forward(p_ang_vel, gt_ang_vel)
                    local d_ang_vel = self.criterion:backward(p_ang_vel, gt_ang_vel):clone()
                    d_ang_vel = d_ang_vel/d_ang_vel:nElement()  -- manually do sizeAverage

                    self.identitycriterion:forward(p_obj_prop, gt_obj_prop)
                    local d_obj_prop = self.identitycriterion:backward(p_obj_prop, gt_obj_prop):clone()

                    local obj_d_pred = splitter:backward({prediction_object}, {d_pos, d_vel, d_ang, d_ang_vel, d_obj_prop}):clone()

                    table.insert(d_pred_objs, obj_d_pred)
                else
                    table.insert(d_pred_objs, convert_type(torch.zeros(mp.batch_size, 1, mp.object_dim), mp.cuda))
                end
            end

            -- -- print('....')
            -- for j=1,#d_pred_objs do
            --     print(d_pred_objs[j]:norm())  -- why are all of them identical? Check this! They are not identical from above though PROBLEM!
            -- end
            -- print(zero_ind)
            -- -- assert(false)
            d_pred_objs = torch.cat(d_pred_objs, 2)

            -- print(d_pred_objs:size())
            -- assert(false)
            -- reshape back
            d_pred_objs = d_pred_objs:reshape(mp.batch_size, mp.seq_length*mp.object_dim)
            table.insert(d_pred, d_pred_objs:clone()) -- don't nee
        else
            -- just give it 0s if we are not at the last timestep
            table.insert(d_pred, convert_type(torch.zeros(mp.batch_size, mp.seq_length*mp.object_dim),mp.cuda))
        end
    end


    -- print(d_pred)
    -- for i=1,#d_pred do
    --     print(d_pred[i]:norm())
    -- end
    -- assert(false)

    -- d_input is nonzero everywhere, meaning that gradients have been accumulated through time.
    local d_input = self.network:backward(input,d_pred)  -- updates grad_params

    -- for i=1,#d_input do
    --     print(d_input[i]:norm())
    -- end


    -- print(input)
    -- assert(false)
    collectgarbage()
    return self.theta.grad_params
end


function model:update_position(this, pred)
    -- this: (mp.batch_size, mp.num_past, mp.object_dim)
    -- prediction: (mp.batch_size, mp.num_future, mp.object_dim)
    -- pred is with respect to this[{{},{-1}}]
    ----------------------------------------------------------------------------
    local px = config_args.si.px
    local py = config_args.si.py
    local vx = config_args.si.vx
    local vy = config_args.si.vy
    local pnc = config_args.position_normalize_constant
    local vnc = config_args.velocity_normalize_constant

    local this, pred = this:clone(), pred:clone()
    local lastpos = (this[{{},{-1},{px,py}}]:clone()*pnc)
    local lastvel = (this[{{},{-1},{vx,vy}}]:clone()*vnc)
    local currpos = (pred[{{},{},{px,py}}]:clone()*pnc)
    local currvel = (pred[{{},{},{vx,vy}}]:clone()*vnc)

    -- this is length n+1
    local pos = torch.cat({lastpos, currpos},2)
    local vel = torch.cat({lastvel, currvel},2)

    -- iteratively update pos through num_future 
    for i = 1,pos:size(2)-1 do
        pos[{{},{i+1},{}}] = pos[{{},{i},{}}] + vel[{{},{i},{}}]  -- last dim=2
    end

    -- normalize again
    pos = pos/pnc
    assert(pos[{{},{1},{}}]:size(1) == pred:size(1))

    pred[{{},{},{px,py}}] = pos[{{},{2,-1},{}}]  -- reassign back to pred
    return pred
end


function model:update_angle(this, pred)
    local a = config_args.si.a
    local av = config_args.si.av
    local anc = config_args.angle_normalize_constant

    local this, pred = this:clone(), pred:clone()

    local last_angle = this[{{},{-1},{a}}]:clone()*anc
    local last_angular_velocity = this[{{},{-1},{av}}]:clone()*anc
    local curr_angle = pred[{{},{},{a}}]:clone()*anc
    local curr_angular_velocity = pred[{{},{},{av}}]:clone()*anc

    -- this is length n+1
    local ang = torch.cat({last_angle, curr_angle},2)
    local ang_vel = torch.cat({last_angular_velocity, curr_angular_velocity},2)

    -- iteratively update ang through time. 
    for i = 1,ang:size(2)-1 do
        ang[{{},{i+1},{}}] = ang[{{},{i},{}}] + ang_vel[{{},{i},{}}]  -- last dim=2
    end

    -- normalize again
    ang = ang/anc
    assert(ang[{{},{1},{}}]:size(1) == pred:size(1))

    pred[{{},{},{a}}] = ang[{{},{2,-1},{}}]  -- reassign back to pred
    return pred
end

-- return a table of euc dist between this and each of context
-- size is the number of items in context
-- is this for the last timestep of this?
-- TODO_lowpriority: later we can plot for all timesteps
function model:get_euc_dist(this, context, t)
    local num_context = context:size(2)
    local t = t or -1  -- default use last timestep
    local px = config_args.si.px
    local py = config_args.si.py

    local this_pos = this[{{},{t},{px, py}}]
    local context_pos = context[{{},{},{t},{px, py}}]
    local euc_dists = self:euc_dist(this_pos:repeatTensor(1,num_context,1), context_pos)
    euc_dists = torch.split(euc_dists, 1,2)  --convert to table of (bsize, 1, 1)
    for i=1,#euc_dists do
        euc_dists[i] = torch.squeeze(euc_dists[i])
    end
    return euc_dists
end

-- b and a must be same size
function model:euc_dist(a,b)
    local diff = torch.squeeze(b - a, 3) -- (bsize, num_context, 2)
    local diffsq = torch.pow(diff,2)
    local euc_dists = torch.sqrt(diffsq[{{},{},{1}}]+diffsq[{{},{},{2}}])  -- (bsize, num_context, 1)
    return euc_dists
end

-- update position at time t to get position at t+1
-- default t is the last t
function model:update_position_one(state, t)
    local t = t or -1
    local px = config_args.si.px
    local py = config_args.si.py
    local vx = config_args.si.vx
    local vy = config_args.si.vy
    local pnc = config_args.position_normalize_constant
    local vnc = config_args.velocity_normalize_constant

    local pos_now, vel_now
    if state:dim() == 4 then
        pos_now = state[{{},{},{t},{px, py}}]
        vel_now = state[{{},{},{t},{vx, vy}}]
    else
        pos_now = state[{{},{t},{px, py}}]
        vel_now = state[{{},{t},{vx, vy}}]
    end

    local pos_next = (pos_now:clone()*pnc + vel_now:clone()*vnc)/pnc
    return pos_next, pos_now
end

-- similar to update_position
function model:get_velocity_direction(this, context, t)
    local num_context = context:size(2)

    local this_pos_next, this_pos_now = self:update_position_one(this)
    local context_pos_next, context_pos_now = self:update_position_one(context)

    -- find difference in distances from this_pos_now to context_pos_now
    -- and from his_pos_now to context_pos_next. This will be +/- number
    local euc_dist_now = self:euc_dist(this_pos_now:repeatTensor(1,num_context,1), context_pos_now)
    local euc_dist_next = self:euc_dist(this_pos_now:repeatTensor(1,num_context,1), context_pos_next)
    local euc_dist_diff = euc_dist_next - euc_dist_now  -- (bsize, num_context, 1)  negative if context moving toward this
    euc_dist_diffs = torch.split(euc_dist_diff, 1,2)  --convert to table of (bsize, 1, 1)
    for i=1,#euc_dist_diffs do
        euc_dist_diffs[i] = torch.squeeze(euc_dist_diffs[i])
    end
    -- assert(false)
    return euc_dist_diffs
end

-- simulate batch forward one timestep
function model:sim(batch)
    
    -- get data
    local this_orig, context_orig, y_orig, context_future_orig, mask = unpack(batch)  -- NOTE CHANGE BATCH HERE

    -- crop to number of timestesp
    y_orig = y_orig[{{},{1, numsteps}}]
    context_future_orig = context_future_orig[{{},{},{1, numsteps}}]

    local num_particles = torch.find(mask,1)[1] + 1

    -- arbitrary notion of ordering here
    -- past: (bsize, num_particles, mp.numpast*mp.objdim)
    -- future: (bsize, num_particles, (mp.winsize-mp.numpast), mp.objdim)
    local past = torch.cat({unsqueeze(this_orig:clone(),2), context_orig},2)
    local future = torch.cat({unsqueeze(y_orig:clone(),2), context_future_orig},2)

    assert(past:size(2) == num_particles and future:size(2) == num_particles)

    local pred_sim = model_utils.transfer_data(
                        torch.zeros(mp.batch_size, num_particles,
                                    numsteps, mp.object_dim),
                        mp.cuda)

    -- loop through time
    for t = 1, numsteps do

        -- for each particle, update to the next timestep, given
        -- the past configuration of everybody
        -- total_particles = total_particles+num_particles

        for j = 1, num_particles do
            -- construct batch
            local this = torch.squeeze(past[{{},{j}}])

            local context
            if j == 1 then
                context = past[{{},{j+1,-1}}]
            elseif j == num_particles then
                context = past[{{},{1,-2}}]
            else
                context = torch.cat({past[{{},{1,j-1}}],
                                                    past[{{},{j+1,-1}}]},2)
            end

            local y = future[{{},{j},{t}}]
            y:resize(mp.batch_size, mp.num_future, mp.object_dim)

            local batch = {this, context, y, _, mask}

            -- predict
            local loss, pred = model:fp(params_,batch,true)   -- NOTE CHANGE THIS!
            avg_loss = avg_loss + loss
            count = count + 1

            pred = pred:reshape(mp.batch_size, mp.num_future, mp.object_dim)
            this = this:reshape(mp.batch_size, mp.num_past, mp.object_dim)  -- unnecessary

            -- -- relative coords for next timestep
            if mp.relative then
                pred = data_process.relative_pair(this, pred, true)
            end

            -- restore object properties because we aren't learning them
            pred[{{},{},{config_args.ossi,-1}}] = this[{{},{-1},{config_args.ossi,-1}}]  -- NOTE! THIS DOESN'T TAKE ANGLE INTO ACCOUNT!
            
            -- update position
            pred = update_position(this, pred)

            -- update angle
            pred = update_angle(this, pred)
            -- pred = unsqueezer:forward(pred)
            pred = unsqueeze(pred, 2)

            -- write into pred_sim
            pred_sim[{{},{j},{t},{}}] = pred
        end

        -- update past for next timestep
        if mp.num_past > 1 then
            past = torch.cat({past[{{},{},{2,-1},{}}],
                                pred_sim[{{},{},{t},{}}]}, 3)
        else
            assert(mp.num_past == 1)
            past = pred_sim[{{},{},{t},{}}]:clone()
        end

        -- local this_orig, context_orig, y_orig, context_future_orig, this_pred, context_future_pred, loss = model:sim(batch)

        
    end
    --- to be honest I don't think we need to break into past and context
    -- future, but actually that might be good for coloriing past and future, but
    -- actually I don't think so. For now let's just adapt it

    -- at this point, pred_sim should be all filled out
    -- break pred_sim into this and context_future
    -- recall: pred_sim: (batch_size,seq_length+1,numsteps,object_dim)
    -- recall that you had defined this_pred as the first obj in the future tensor
    local this_pred = torch.squeeze(pred_sim[{{},{1}}])
    if numsteps == 1 then this_pred = unsqueeze(this_pred,2) end

    local context_pred = pred_sim[{{},{2,-1}}]

    if mp.relative then
        y_orig = data_process.relative_pair(this_orig, y_orig, true)
    end

    -- local this_orig, context_orig, y_orig, context_future_orig, this_pred, context_future_pred, loss = model:sim(batch)

    -- if saveoutput and i <= mp.ns then
    --     save_ex_pred_json({this_orig, context_orig,
    --                         y_orig, context_future_orig,
    --                         this_pred, context_pred},
    --                         'batch'..test_loader.datasamplers[current_dataset].current_sampled_id..'.json',
    --                         current_dataset)
    -- end
end
-- print('LOADED ME')
return model
