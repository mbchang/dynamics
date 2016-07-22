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
    local num_layers = params.layers+2


    local step = nn.Sequential()
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
            self.network:float()
            self.criterion:float()
            self.identitycriterion:float()
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

function model:unpack_batch(batch, sim)
    local this, context, this_future, context_future, mask = unpack(batch)
    local past = torch.cat({unsqueeze(this:clone(),2), context},2)
    local future = torch.cat({unsqueeze(this_future:clone(),2), context_future},2)

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

    print(prediction)
    print(torch.squeeze(prediction[1]))
    print(torch.squeeze(prediction[2]))
    assert(false)

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
    end
    loss = loss/#prediction

    collectgarbage()
    return loss, prediction
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
        d_vel = d_vel/d_vel:nElement()  -- manually do sizeAverage

        self.identitycriterion:forward(p_ang, gt_ang)
        local d_ang = self.identitycriterion:backward(p_ang, gt_ang):clone()

        self.criterion:forward(p_ang_vel, gt_ang_vel)
        local d_ang_vel = self.criterion:backward(p_ang_vel, gt_ang_vel):clone()
        d_ang_vel = d_ang_vel/d_ang_vel:nElement()  -- manually do sizeAverage

        self.identitycriterion:forward(p_obj_prop, gt_obj_prop)
        local d_obj_prop = self.identitycriterion:backward(p_obj_prop, gt_obj_prop):clone()

        local obj_d_pred = splitter:backward({prediction[i]}, {d_pos, d_vel, d_ang, d_ang_vel, d_obj_prop})

        table.insert(d_pred, obj_d_pred)
    end

    self.network:backward(all_past,d_pred)  -- updates grad_params

    collectgarbage()
    return self.theta.grad_params
end

function model:sim(batch, numsteps)

end

return model
