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
    -- input: (bsize, num_obj*num_past*obj_dim)
    -- output: (bsize, num_obj*num_future*obj_dim)
    -- hiddim: (bsize, num_obj*rnn_dim)
    local rnn_dim = params.rnn_dim
    local obj_dim = params.object_dim
    local max_obj = params.seq_length
    local num_past = params.num_past
    local num_future = params.num_future
    local in_dim = max_obj*num_past*obj_dim
    local hid_dim = max_obj*rnn_dim  -- TODO rename rnn_dim to hid_dim
    local out_dim = max_obj*num_future*obj_dim  -- note that we will be ignoring the padded areas during backpropagation
    local num_layers = params.layers+2

    local net = nn.Sequential()
    for i = 1, num_layers do -- TODO make sure this is comparable to encoder decoder architecture in terms of layers
        if i == 1 then 
            net:add(nn.Linear(in_dim, hid_dim))
        elseif i == num_layers then 
            net:add(nn.Linear(hid_dim, out_dim))
        else
            net:add(nn.Linear(hid_dim, hid_dim))
        end
        net:add(nn.ReLU())  
        if mp.batch_norm then 
            net:add(nn.BatchNormalization(hid_dim))
        end
    end
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

function model:pad(tensor, dim, num_to_pad)
    local tensor_size = torch.totable(tensor:size())
    assert(dim>=1 and dim<=#tensor_size)

    tensor_size[dim] = num_to_pad
    local padding = torch.zeros(unpack(tensor_size))
    if self.mp.cuda then padding = padding:cuda() end

    local padded = torch.cat({tensor, padding}, dim)
    return padded
end

function model:unpack_batch(batch, sim)
    -- past is reshaped, future is not
    local this, context, this_future, context_future, mask = unpack(batch)
    local past = torch.cat({unsqueeze(this:clone(),2), context},2)
    local future = torch.cat({unsqueeze(this_future:clone(),2), context_future},2)

    -- YOU HAVE TO PAD!!!!!
    local max_obj = self.mp.seq_length -- TODO: HARDCODED
    local bsize, num_obj = past:size(1), past:size(2)
    local num_past, num_future = past:size(3), future:size(3)
    local obj_dim = past:size(4)

    past = self:pad(past, 2, max_obj-num_obj)
    -- future = self:pad(future, 2, max_obj-num_obj)

    past:resize(bsize, max_obj*num_past*obj_dim)
    -- future:resize(bsize, max_obj*num_future*obj_dim)

    return past, future
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

    local past, future = self:unpack_batch(batch, sim)
    local prediction = self.network:forward(past)
    local future_size = torch.totable(future:size())
    local num_obj = future_size[2]
    future_size[2] = self.mp.seq_length
    prediction = prediction:reshape(unpack(future_size))
    prediction = prediction[{{},{1,num_obj}}]  -- ignore padding

    -- now you can compare future and prediction: (bsize, num_obj, num_future, obj_dim)
    local p_vel = prediction[{{},{},{},{config_args.si.vx,config_args.si.vy}}]
    local gt_vel = future[{{},{},{},{config_args.si.vx,config_args.si.vy}}]

    local p_ang_vel = prediction[{{},{},{},{config_args.si.av}}]
    local gt_ang_vel = future[{{},{},{},{config_args.si.av}}]

    local loss_vel = self.criterion:forward(p_vel, gt_vel)
    local loss_ang_vel = self.criterion:forward(p_ang_vel, gt_ang_vel)
    local loss = loss_vel + loss_ang_vel
    loss = loss/(p_vel:nElement()+p_ang_vel:nElement()) -- manually do size average

    collectgarbage()
    return loss, prediction
end


-- local p_pos, p_vel, p_obj_prop=split_output(params):forward(prediction)
-- local gt_pos, gt_vel, gt_obj_prop=split_output(params):forward(this_future)
-- a lot of instantiations of split_output
-- prediction: (bsize, num_obj, num_future, obj_dim)
function model:bp(batch, prediction, sim)
    self.theta.grad_params:zero() -- the d_parameters
    local past, future = self:unpack_batch(batch, sim)

    local p_vel = prediction[{{},{},{},{config_args.si.vx,config_args.si.vy}}]
    local gt_vel = future[{{},{},{},{config_args.si.vx,config_args.si.vy}}]

    local p_ang_vel = prediction[{{},{},{},{config_args.si.av}}]
    local gt_ang_vel = future[{{},{},{},{config_args.si.av}}]

    self.criterion:forward(p_vel, gt_vel)
    self.criterion:forward(p_ang_vel, gt_ang_vel)

    local d_vel = self.criterion:backward(p_vel, gt_vel):clone()
    d_vel = d_vel/d_vel:nElement()  -- manually do sizeAverage

    local d_ang_vel = self.criterion:backward(p_ang_vel, gt_ang_vel):clone()
    d_ang_vel = d_ang_vel/d_ang_vel:nElement()  -- manually do sizeAverage

    -- the things we don't want to back prop through
    local bsize, num_obj, num_future, obj_dim = unpack(torch.totable(future:size()))
    local d_pos = convert_type(torch.zeros(bsize, num_obj, num_future, 2), self.mp.cuda)
    local d_ang = convert_type(torch.zeros(bsize, num_obj, num_future, 1), self.mp.cuda)
    local d_obj_prop = convert_type(torch.zeros(bsize, num_obj, num_future, obj_dim-config_args.si.m[1]+1), self.mp.cuda)
    
    -- combine them all
    local d_pred = torch.cat({d_pos, d_vel, d_ang, d_ang_vel, d_obj_prop},4)

    -- pad again
    d_pred = self:pad(d_pred, 2, mp.seq_length-num_obj)
    d_pred = d_pred:reshape(bsize, mp.seq_length, num_future, obj_dim)
    self.network:backward(past, d_pred)

    collectgarbage()
    return self.theta.grad_params
end

return model
