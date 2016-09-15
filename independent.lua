require 'nn'
require 'rnn'
require 'torch'
require 'nngraph'
require 'Base'
require 'IdentityCriterion'
require 'data_utils'
require 'modules'

nngraph.setDebug(true)

-- with a bidirectional lstm, no need to put a mask
-- however, you can have variable sequence length now!
function init_network(params)
    -- input: table of length num_obj with size (bsize, num_past*obj_dim)
    -- output: table of length num_obj with size (bsize, num_future*obj_dim)
    local hid_dim = params.rnn_dim
    local obj_dim = params.object_dim
    local max_obj = params.seq_length
    local num_past = params.num_past
    local num_future = params.num_future
    local in_dim = num_past*obj_dim
    local out_dim = num_future*obj_dim  -- note that we will be ignoring the padded areas during backpropagation
    local num_layers = params.layers

    assert(num_layers > 0)

    local step = nn.Sequential()

    if num_layers == 1 then
        step:add(nn.Linear(in_dim, out_dim))
    else
        for i = 1,num_layers do
            if i == 1 then 
                step:add(nn.Linear(in_dim, hid_dim))
                step:add(nn.ReLU())
            elseif i == num_layers then 
                step:add(nn.Linear(hid_dim, out_dim))
                step:add(nn.Reshape(num_future, obj_dim))
            else
                step:add(nn.Linear(hid_dim, hid_dim))
                step:add(nn.ReLU())
            end
            if mp.batch_norm then 
                step:add(nn.BatchNormalization(params.rnn_dim))
            end
        end
    end

    local net = nn.Sequencer(step)
    net:remember('neither')

    return net
end


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

function model:cuda()
    self.network:cuda()
    self.criterion:cuda()
    self.identitycriterion:cuda()
end

function model:float()
    self.network:float()
    self.criterion:float()
    self.identitycriterion:float()
end

function model:clearState()
    self.network:clearState()
end

function model:unpack_batch(batch, sim)
    local this, context, this_future, context_future, mask = unpack(batch)

    -- need to do relative pair
    context_future[{{},{},{},{1,6}}] = context_future[{{},{},{},{1,6}}] - context[{{},{},{-1},{1,6}}]:expandAs(context_future[{{},{},{},{1,6}}])

    local past = torch.cat({unsqueeze(this:clone(),2), context},2)
    local future = torch.cat({unsqueeze(this_future:clone(),2), context_future},2)

    -- print(this_future:norm())

    local bsize, num_obj = past:size(1), past:size(2)
    local num_past, num_future = past:size(3), future:size(3)
    local obj_dim = past:size(4)

    -- now break into different trajectories
    local all_past = {}
    local all_future = {}
    for i =1,num_obj do
        table.insert(all_past,past[{{},{i}}]:reshape(bsize,num_past*obj_dim))
        table.insert(all_future,future[{{},{i}}]:reshape(bsize,num_future*obj_dim))
    end

    return all_past, all_future
end

-- Input to fp
-- {
--   1 : DoubleTensor - size: 4x2x9
--   2 : DoubleTensor - size: 4x2x2x9
--   3 : DoubleTensor - size: 4x48x9
--   4 : DoubleTensor - size: 4x2x48x9
--   5 : DoubleTensor - size: 10
-- }
function model:fp(params_, batch, sim)
    if params_ ~= self.theta.params then self.theta.params:copy(params_) end
    self.theta.grad_params:zero()  -- reset gradient

    local all_past, all_future = self:unpack_batch(batch, sim)
    local prediction = self.network:forward(all_past)

    local loss_vels = 0
    local loss_ang_vels = 0

    local loss = 0
    for i = 1,#prediction do
        -- table of length num_obj of {bsize, num_future, obj_dim}
        local p_pos, p_vel, p_ang, p_ang_vel, p_obj_prop =
                            unpack(split_output(self.mp):forward(prediction[i]))
        local gt_pos, gt_vel, gt_ang, gt_ang_vel, gt_obj_prop =
                            unpack(split_output(self.mp):forward(all_future[i]))

        local loss_vel = self.criterion:forward(p_vel, gt_vel)
        local loss_ang_vel = self.criterion:forward(p_ang_vel, gt_ang_vel)
        local obj_loss = loss_vel + loss_ang_vel
        obj_loss = obj_loss/(p_vel:nElement()+p_ang_vel:nElement()) -- manually do size average
        loss = loss + obj_loss

        loss_vels = loss_vels + loss_vel/p_vel:nElement()
        loss_ang_vels = loss_ang_vels + loss_ang_vel/p_ang_vel:nElement()

    end
    loss = loss/#prediction

    collectgarbage()
    return loss, prediction, loss_vels, loss_ang_vels
end

function model:fp_batch(params_, batch, sim)
    if params_ ~= self.theta.params then self.theta.params:copy(params_) end
    self.theta.grad_params:zero()  -- reset gradient

    local all_past, all_future = self:unpack_batch(batch, sim)

    local prediction = self.network:forward(all_past)

    -- predict for only the first object (which is this)
    local j = 1  -- this is the index of this
    -- table of length num_obj of {bsize, num_future, obj_dim}
    local p_pos, p_vel, p_ang, p_ang_vel, p_obj_prop =
                        unpack(split_output(self.mp):forward(prediction[j]))
    local gt_pos, gt_vel, gt_ang, gt_ang_vel, gt_obj_prop =
                        unpack(split_output(self.mp):forward(all_future[j]))

    local loss_all = {}
    for i=1,mp.batch_size do
        local loss_vel = self.criterion:forward(p_vel[{{i}}], gt_vel[{{i}}])
        local loss_ang_vel = self.criterion:forward(p_ang_vel[{{i}}], gt_ang_vel[{{i}}])
        local loss = loss_vel + loss_ang_vel
        loss = loss/(p_vel[{{i}}]:nElement()+p_ang_vel[{{i}}]:nElement()) -- manually do size average
        table.insert(loss_all, loss)
    end

    collectgarbage()
    return torch.Tensor(loss_all), prediction
end


-- local p_pos, p_vel, p_obj_prop=split_output(params):forward(prediction)
-- local gt_pos, gt_vel, gt_obj_prop=split_output(params):forward(this_future)
-- a lot of instantiations of split_output
function model:bp(batch, prediction, sim)
    self.theta.grad_params:zero() -- the d_parameters
    local all_past, all_future = self:unpack_batch(batch, sim)

    local splitter = split_output(self.mp)

    local d_pred = {}
    for i = 1, #prediction do

        local p_pos, p_vel, p_ang, p_ang_vel, p_obj_prop = unpack(splitter:forward(prediction[i]))
        local gt_pos, gt_vel, gt_ang, gt_ang_vel, gt_obj_prop =
                            unpack(split_output(self.mp):forward(all_future[i]))

        -- NOTE! is there a better loss function for angle?
        self.identitycriterion:forward(p_pos, gt_pos)
        local d_pos = self.identitycriterion:backward(p_pos, gt_pos):clone()

        self.criterion:forward(p_vel, gt_vel)
        local d_vel = self.criterion:backward(p_vel, gt_vel):clone()
        d_vel:mul(mp.vlambda)
        d_vel = d_vel/d_vel:nElement()  -- manually do sizeAverage

        self.identitycriterion:forward(p_ang, gt_ang)
        local d_ang = self.identitycriterion:backward(p_ang, gt_ang):clone()

        self.criterion:forward(p_ang_vel, gt_ang_vel)
        local d_ang_vel = self.criterion:backward(p_ang_vel, gt_ang_vel):clone()
        d_ang_vel:mul(mp.lambda)
        d_ang_vel = d_ang_vel/d_ang_vel:nElement()  -- manually do sizeAverage

        self.identitycriterion:forward(p_obj_prop, gt_obj_prop)
        local d_obj_prop = self.identitycriterion:backward(p_obj_prop, gt_obj_prop):clone()

        local obj_d_pred = splitter:backward({prediction[i]}, {d_pos, d_vel, d_ang, d_ang_vel, d_obj_prop}):clone()

        table.insert(d_pred, obj_d_pred)
    end

    self.network:backward(all_past,d_pred)  -- updates grad_params

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
    return compute_euc_dist(a,b)
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


return model
