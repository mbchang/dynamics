require 'nn'
require 'torch'
require 'nngraph'
require 'Base'
local model_utils = require 'model_utils'

nngraph.setDebug(true)

function init_object_encoder(input_dim, rnn_inp_dim)
    assert(rnn_inp_dim % 2 == 0)
    local thisp     = nn.Identity()() -- this particle of interest  (batch_size, input_dim)
    local contextp  = nn.Identity()() -- the context particle  (batch_size, partilce_dim)

    local thisp_out     = nn.ReLU()(nn.Linear(input_dim, rnn_inp_dim/2)(thisp))  -- (batch_size, rnn_inp_dim/2)
    local contextp_out  = nn.ReLU()(nn.Linear(input_dim, rnn_inp_dim/2)(contextp)) -- (batch_size, rnn_inp_dim/2)

    -- Concatenate
    local encoder_out = nn.JoinTable(2)({thisp_out, contextp_out})  -- (batch_size, rnn_inp_dim)

    return nn.gModule({thisp, contextp}, {encoder_out})
end


function init_object_decoder(rnn_hid_dim, num_future, object_dim)
    local rnn_out = nn.Identity()()  -- rnn_out had better be of dim (batch_size, rnn_hid_dim)

    local out_dim = num_future * object_dim
    -- -- ok, here we will have to split up the output
    local world_state_pre, obj_prop_pre = split_tensor(3,{num_future, object_dim})(nn.Linear(rnn_hid_dim, out_dim)(rnn_out)):split(2)
    local obj_prop = nn.Sigmoid()(obj_prop_pre)
    local world_state = world_state_pre -- linear
    local dec_out_reshaped = nn.JoinTable(3)({world_state, obj_prop})
    local decoder_out = nn.Reshape(out_dim, true)(dec_out_reshaped)

    return nn.gModule({rnn_out}, {decoder_out})
end

function init_feedforward(rnn_hid_dim, numlayers)
    local input = nn.Identity()()
    local ff = nn.Sequential()
    for i = 1,numlayers do
        ff:add(nn.Linear(rnn_hid_dim, rnn_hid_dim))
        ff:add(nn.Sigmoid())
    end
    local output = ff(input)
    return nn.gModule({input}, {output})
end

-- reshape is something like (10,8)
-- the numbers are hardcoded
-- Takes a tensor, and returns its two halves, split on the particular dim
-- For example, give  a tensor of size (260, 10, 8), if dim is 3, then
-- this returns a table of two tensors: {(260,10,4), (260, 10,4)}
function split_tensor(dim, reshape)
    assert(reshape[2] %2 == 0)
    local tensor = nn.Identity()()
    local reshaped = nn.Reshape(reshape[1],reshape[2], 1, true)(tensor)
    local splitted = nn.SplitTable(dim)(reshaped)
    local first_half = nn.JoinTable(dim)(nn.NarrowTable(1,reshape[2]/2)(splitted))
    local second_half = nn.JoinTable(dim)(nn.NarrowTable(1+reshape[2]/2,reshape[2]/2)(splitted))
    return nn.gModule({tensor},{first_half, second_half})
end

-- do not need the mask
-- params: layers, input_dim, goo_dim, rnn_inp_dim, rnn_hid_dim, out_dim
function init_network(params)
    -- Initialize encoder and decoder
    local encoder = init_object_encoder(params.input_dim, params.rnn_dim)
    local decoder = init_object_decoder(params.rnn_dim, params.num_future, params.object_dim)
    local ff = init_feedforward(params.rnn_dim, params.layers)

    -- Input to netowrk
    local thisp_past    = nn.Identity()() -- this particle of interest, past
    local contextp      = nn.Identity()() -- the context particle
    local thisp_future  = nn.Identity()() -- the particle of interet, future

    -- Input to LSTM
    -- actually can replace all of this with karpathy lstm
    local model_input = encoder({thisp_past, contextp})
    local model_output = ff(model_input)
    local prediction = decoder({model_output})  -- next_h is the output of the last layer

    -- -- split criterion: I know that this works
    local world_state, obj_prop = split_tensor(3, {params.num_future,params.object_dim})({prediction}):split(2)
    local fworld_state, fobj_prop = split_tensor(3, {params.num_future,params.object_dim})({thisp_future}):split(2)

    local err1 = nn.SmoothL1Criterion()({world_state, fworld_state})
    local err2 = nn.BCECriterion()({obj_prop, fobj_prop})
    local err = nn.MulConstant(0.5)(nn.CAddTable()({err1,err2}))  -- it should be the average err

    return nn.gModule({thisp_past, contextp, thisp_future}, {err, prediction})  -- last output should be prediction
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
        self.network = torch.load(model_path):clone()
    else
        self.network = init_network(self.mp)
    end
    if self.mp.cuda then self.network:cuda() end

    self.theta = {}
    self.theta.params, self.theta.grad_params = self.network:getParameters()

    collectgarbage()
    return self
end


function model:fp(params_, x, y)
    if params_ ~= self.theta.params then self.theta.params:copy(params_) end
    self.theta.grad_params:zero()  -- reset gradient

    -- unpack inputs
    local this_past     = model_utils.transfer_data(x.this:clone(), self.mp.cuda)
    local context       = model_utils.transfer_data(x.context:clone(), self.mp.cuda)
    local this_future   = model_utils.transfer_data(y:clone(), self.mp.cuda)

    assert(this_past:size(1) == self.mp.batch_size and this_past:size(2) == self.mp.input_dim)
    assert(context:size(1) == self.mp.batch_size and context:size(2)==self.mp.seq_length
            and context:size(3) == self.mp.input_dim)
    assert(this_future:size(1) == self.mp.batch_size and this_future:size(2) == self.mp.out_dim)

    -- it makes sense to zero the loss here
    local loss, prediction = unpack(self.network:forward({this_past, context[{{},1}], this_future})) -- because it is padded with 0s elsewhere

    collectgarbage()
    return loss:sum(), prediction  -- we sum the losses through time!  -- the loss:sum() just converts the doubleTensor into a number
end


function model:bp(x, y, mask)
    self.theta.grad_params:zero() -- the d_parameters

    -- unpack inputs. All of these have been CUDAed already if need be
    local this_past     = model_utils.transfer_data(x.this:clone(), self.mp.cuda)
    local context       = model_utils.transfer_data(x.context:clone(), self.mp.cuda)
    local this_future   = model_utils.transfer_data(y:clone(), self.mp.cuda)

    assert(mask[1] == 1)
    local derr = model_utils.transfer_data(torch.ones(1), self.mp.cuda)
    local dpred = model_utils.transfer_data(torch.zeros(self.mp.batch_size,self.mp.out_dim), self.mp.cuda)
    local dtp, dc, dtf = unpack(self.network:backward({this_past, context[{{},i}], this_future},{derr, dpred}))
    self.theta.grad_params:clamp(-self.mp.max_grad_norm, self.mp.max_grad_norm)
    collectgarbage()
    return self.theta.grad_params
end

return model
