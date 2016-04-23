require 'nn'
require 'torch'
require 'nngraph'
require 'Base'
local model_utils = require 'model_utils'

nngraph.setDebug(true)

function lstm(x, prev_c, prev_h, params)
    -- Calculate all four gates in one go
    local i2h = nn.Linear(params.rnn_dim, 4*params.rnn_dim)(x) -- this is the problem when I go two layers
    local h2h = nn.Linear(params.rnn_dim, 4*params.rnn_dim)(prev_h)
    local gates = nn.CAddTable()({i2h, h2h})

    -- Reshape to (bsize, n_gates, hid_size)
    -- Then slize the n_gates dimension, i.e dimension 2
    local reshaped_gates =  nn.Reshape(4,params.rnn_dim)(gates)
    local sliced_gates = nn.SplitTable(2)(reshaped_gates)

    -- Use select gate to fetch each gate and apply nonlinearity
    local in_gate          = nn.Sigmoid()(nn.SelectTable(1)(sliced_gates))
    local in_transform     = nn.Tanh()(nn.SelectTable(2)(sliced_gates))
    local forget_gate      = nn.Sigmoid()(nn.SelectTable(3)(sliced_gates))
    local out_gate         = nn.Sigmoid()(nn.SelectTable(4)(sliced_gates))

    local next_c           = nn.CAddTable()({
      nn.CMulTable()({forget_gate, prev_c}),
      nn.CMulTable()({in_gate,     in_transform})
    })
    local next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

    return next_c, next_h
end


function init_object_encoder(input_dim, rnn_inp_dim)
    assert(rnn_inp_dim % 2 == 0)
    local thisp     = nn.Identity()() -- (batch_size, input_dim)
    local contextp  = nn.Identity()() -- (batch_size, partilce_dim)

    -- (batch_size, rnn_inp_dim/2)
    local thisp_out     = nn.ReLU()
                            (nn.Linear(input_dim, rnn_inp_dim/2)(thisp))
    local contextp_out  = nn.ReLU()
                            (nn.Linear(input_dim, rnn_inp_dim/2)(contextp))

    -- Concatenate
    -- (batch_size, rnn_inp_dim)
    local encoder_out = nn.JoinTable(2)({thisp_out, contextp_out})

    return nn.gModule({thisp, contextp}, {encoder_out})
end


function init_object_decoder(rnn_hid_dim, num_future, object_dim)
    -- rnn_out had better be of dim (batch_size, rnn_hid_dim)
    local rnn_out = nn.Identity()()
    -- local decoder_out = nn.Tanh()(nn.Linear(rnn_hid_dim, out_dim)(rnn_out))
    local out_dim = num_future * object_dim
    local decoder_preout = nn.Linear(rnn_hid_dim, out_dim)(rnn_out)

    -- -- ok, here we will have to split up the output
    local world_state_pre, obj_prop_pre = split_tensor(3,
                {num_future, object_dim},{{1,4},{5,object_dim}})
                (decoder_preout):split(2)   -- contains info about object dim!
    local obj_prop = nn.Sigmoid()(obj_prop_pre)
    local world_state = world_state_pre -- linear
    local dec_out_reshaped = nn.JoinTable(3)({world_state, obj_prop})
    local decoder_out = nn.Reshape(out_dim, true)(dec_out_reshaped)
    return nn.gModule({rnn_out}, {decoder_out})
end


-- boundaries: {{l1,r1},{l2,r2},{l3,r3},etc}
function split_tensor(dim, reshape, boundaries)
    -- assert(reshape[2] %2 == 0)
    local tensor = nn.Identity()()
    local reshaped = nn.Reshape(reshape[1],reshape[2], 1, true)(tensor)
    local splitted = nn.SplitTable(dim)(reshaped)
    local chunks = {}
    for cb = 1,#boundaries do
        local left,right = unpack(boundaries[cb])
        chunks[#chunks+1] = nn.JoinTable(dim)(nn.NarrowTable(left,right)(splitted))
    end
    return nn.gModule({tensor},chunks)
end

-- do not need the mask
-- params: layers, input_dim, goo_dim, rnn_inp_dim, rnn_hid_dim, out_dim
function init_network(params)
    -- Initialize encoder and decoder
    local encoder = init_object_encoder(params.input_dim, params.rnn_dim)
    local decoder = init_object_decoder(params.rnn_dim, params.num_future, params.object_dim)

    -- Input to netowrk
    local thisp_past    = nn.Identity()() -- this particle of interest, past
    local contextp      = nn.Identity()() -- the context particle
    local thisp_future  = nn.Identity()() -- the particle of interet, future

    -- Input to LSTM
    -- actually can replace all of this with karpathy lstm
    local lstm_input = encoder({thisp_past, contextp})
    local prev_s = nn.Identity()() -- LSTM

    -- Go through each layer of LSTM
    local rnn_inp = {[0] = nn.Identity()(lstm_input)}  -- rnn_inp[i] holds the input at layer i+1
    local next_s = {}
    local split = {prev_s:split(2 * params.layers)}
    local next_c, next_h
    for layer_idx = 1, params.layers do
        local prev_c         = split[2 * layer_idx - 1]  -- odd entries
        local prev_h         = split[2 * layer_idx]  -- even entries
        local dropped        = rnn_inp[layer_idx - 1]
        next_c, next_h = lstm(dropped, prev_c, prev_h, params)  -- you can make this a gModule if you wnant
        table.insert(next_s, next_c)
        table.insert(next_s, next_h)
        rnn_inp[layer_idx] = next_h
    end
    --
    -- local klstm = LSTM.lstm(params.rnn_dim, params.rnn_dim, params.layers, 0)
    -- local outputs = klstm({lstm_input, unpack(prev_s)})

    local prediction = decoder({next_h})  -- next_h is the output of the last layer

    -- -- split criterion: I know that this works
    local world_state, obj_prop = split_tensor(3,
        {params.num_future,params.object_dim},{{1,4},{5,params.object_dim}})
        ({prediction}):split(2)
    local gtworld_state, gtobj_prop = split_tensor(3,
        {params.num_future,params.object_dim},{{1,4},{5,params.object_dim}})
        ({thisp_future}):split(2)

    -- split state: only pass gradients on velocity
    local pos, vel = split_tensor(3,
        {params.num_future, POSVELDIM},{{1,2},{3,4}})
        ({world_state}):split(2) -- split world_state in half on last dim
    local gt_pos, gt_vel = split_tensor(3,
        {params.num_future, POSVELDIM},{{1,2},{3,4}})
        ({gtworld_state}):split(2)

    -- don't pass gradients on position or object properties
    -- MSE Criterion for velocity
    local err0 = nn.IdentityCriterion()({pos, gt_pos})
    local err1 = nn.MSECriterion()({vel, gt_vel})
    local err2 = nn.IdentityCriterion()({obj_prop, gtobj_prop})
    local err = nn.MulConstant(1)(nn.CAddTable()({err0,err1,err2}))

    return nn.gModule({thisp_past, contextp, thisp_future},
                    {err, prediction})  -- last output should be prediction
end


--------------------------------------------------------------------------------
--############################################################################--
--------------------------------------------------------------------------------

-- Now create the model class
local model = {}
model.__index = model

-- like Tejas's init() method
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
    else
        self.network = init_network(self.mp)
        if self.mp.cuda then self.network:cuda() end
    end

    self.theta = {}
    self.theta.params, self.theta.grad_params = self.network:getParameters()

    -- This will cache the values that s takes on in one forward pass
    self.s = {}
    for j = 0, self.mp.seq_length do
        self.s[j] = {}
        for d = 1, 2 * self.mp.layers do
            self.s[j][d] = convert_type(torch.zeros(self.mp.batch_size, self.mp.rnn_dim), self.mp.cuda)
        end
    end

    -- This will cache the values of the grad of the s
    self.ds = {}
    for d = 1, 2 * self.mp.layers do
        self.ds[d] = convert_type(torch.zeros(self.mp.batch_size, self.mp.rnn_dim), self.mp.cuda)
    end

    self.rnns = model_utils.clone_many_times(self.network,
                                                self.mp.seq_length,
                                                not self.network.parameters)
    collectgarbage()
    return self
end


function model:reset_state(s, seq_len, num_layers)
    for j = 0, seq_len do
        for d = 1, 2 * num_layers do
            s[j][d]:zero()
        end
    end
    return s
end


function model:reset_ds(ds)
    for d = 1, #self.ds do
        ds[d]:zero()
    end
    return ds
end


function model:fp(params_, batch, sim)
    if params_ ~= self.theta.params then self.theta.params:copy(params_) end
    self.theta.grad_params:zero()  -- reset gradient
    self.s = self:reset_state(self.s, self.mp.seq_length, self.mp.layers) -- because we are doing a fresh new forward pass

    local this, context, y, mask = unpack(batch)
    local x = {this=this,context=context}

    if not sim then
        y = crop_future(y, {y:size(1), mp.winsize-mp.num_past, mp.object_dim},
                            {2,mp.num_future})
    end

    -- unpack inputs
    local this_past     = convert_type(x.this:clone(), self.mp.cuda)
    local context       = convert_type(x.context:clone(), self.mp.cuda)
    local this_future   = convert_type(y:clone(), self.mp.cuda)

    assert(this_past:size(1) == self.mp.batch_size and
            this_past:size(2) == self.mp.input_dim)
    assert(context:size(1) == self.mp.batch_size and
            context:size(2)==self.mp.seq_length
            and context:size(3) == self.mp.input_dim)
    assert(this_future:size(1) == self.mp.batch_size and
            this_future:size(2) == self.mp.out_dim)

    -- it makes sense to zero the loss here
    local loss = convert_type(torch.zeros(self.mp.seq_length), self.mp.cuda)
    local preds = {}
    for i = 1, self.mp.seq_length do
        local sim1 = self.s[i-1]  -- had been reset to 0 for initial pass
        -- Note that I'm feeding in this_future every time
        loss[i], self.s[i], preds[i] = unpack(self.rnns[i]:
            forward({this_past, context[{{},i}], sim1, this_future}))

    end

    local pred = preds[torch.find(mask,1)[1]] -- (1, num_future)

    collectgarbage()
    return loss:sum(), pred -- we sum the losses through time!

end


function model:bp(batch)
    local state = self.s

    self.theta.grad_params:zero() -- the d_parameters
    self.ds = self:reset_ds(self.ds)  -- the d_outputs of the states

    local this, context, y, mask = unpack(batch)
    local x = {this=this,context=context}

    -- unpack inputs. All of these have been CUDAed already if need be
    local this_past     = convert_type(x.this:clone(), self.mp.cuda)
    local context       = convert_type(x.context:clone(), self.mp.cuda)
    local this_future   = convert_type(y:clone(), self.mp.cuda)

    for i = self.mp.seq_length, 1, -1 do  -- go backwards in time
        local sim1 = state[i - 1]
        local derr
        if mask[i] == 1 then
            derr = convert_type(torch.ones(1), self.mp.cuda)
        elseif mask[i] == 0 then
            derr = convert_type(torch.zeros(1), self.mp.cuda)
        else
            error('invalid mask')
        end
        local dpred = convert_type(torch.zeros(
                            self.mp.batch_size,self.mp.out_dim), self.mp.cuda)  -- TODO is this correct?
        local dtp, dc, dsim1, dtf = unpack(self.rnns[i]:backward(
                            {this_past, context[{{},i}], sim1, this_future},
                            {derr, self.ds, dpred}))
        g_replace_table(self.ds, dsim1)
    end
    collectgarbage()
    return self.theta.grad_params
end

return model
