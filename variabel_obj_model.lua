require 'nn'
require 'rnn'
require 'torch'
require 'nngraph'
require 'Base'
require 'IdentityCriterion'
require 'data_utils'

nngraph.setDebug(true)

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

    local out_dim = num_future * object_dim
    -- -- ok, here we will have to split up the output
    local decoder_preout = nn.Linear(rnn_hid_dim, out_dim)(rnn_out)

    if mp.accel then
        local pos_vel_pre, accel_prop_pre = split_tensor(3,
                    {num_future, object_dim},{{1,4},{5,object_dim+2}})
                    (decoder_preout):split(2)
        local accel_prop = nn.Sigmoid()(accel_prop_pre)
        local pos_vel = pos_vel_pre
        local dec_out_reshaped = nn.JoinTable(3)({pos_vel, accel_prop})
        local decoder_out = nn.Reshape(out_dim, true)(dec_out_reshaped)
        return nn.gModule({rnn_out}, {decoder_out})
    else
        local world_state_pre, obj_prop_pre = split_tensor(3,
                    {num_future, object_dim},{{1,4},{5,object_dim}})
                    (decoder_preout):split(2)  -- contains info about objectdim!
        local obj_prop = nn.Sigmoid()(obj_prop_pre)
        local world_state = world_state_pre -- linear
        local dec_out_reshaped = nn.JoinTable(3)({world_state,obj_prop})
        local decoder_out = nn.Reshape(out_dim, true)(dec_out_reshaped)
        return nn.gModule({rnn_out}, {decoder_out})
    end
end

-- with a bidirectional lstm, no need to put a mask
-- however, you can have variable sequence length now!
function init_network(params)
    -- encoder produces: (bsize, rnn_inp_dim)
    -- decoder expects (bsize, 2*rnn_hid_dim)
    local encoder = init_object_encoder(params.input_dim, params.rnn_dim)
    local decoder = init_object_decoder(2*params.rnn_dim, params.num_future,
                                                            params.object_dim)

    local net = nn.Sequential()

    -- note you can just replace LSTM with Linear!
    local lstm_step = nn.Sequential()
    lstm_step:add(encoder)
    lstm_step:add(nn.LSTM(params.rnn_dim,params.rnn_dim))
    lstm_step:add(nn.ReLU())
    for i = 2,params.layers do
        lstm_step:add(nn.LSTM(params.rnn_dim,params.rnn_dim))
        lstm_step:add(nn.ReLU())
    end
    net:add(nn.BiSequencer(lstm_step))
    -- input table of (bsize, 2*d_hid) of length seq_length
    -- output: tensor (bsize, 2*d_hid)
    net:add(nn.CAddTable())  -- add across the "timesteps" to sum contributions
    net:add(decoder)
    return net
end


function split_output(params)
    local POSVELDIM = 4
    local future = nn.Identity()()

    local world_state, obj_prop = split_tensor(3,
        {params.num_future,params.object_dim},{{1,4},{5,params.object_dim}})
        ({future}):split(2)

    -- split state: only pass gradients on velocity
    local pos, vel = split_tensor(3,
        {params.num_future, POSVELDIM},{{1,2},{3,4}})
        ({world_state}):split(2) -- split world_state in half on last dim
    return nn.gModule({future},{pos, vel, obj_prop})
end

-- boundaries: {{l1,r1},{l2,r2},{l3,r3},etc}
function split_tensor(dim, reshape, boundaries)
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
    else
        self.network = init_network(self.mp)
         if self.mp.cuda then self.network:cuda() end
    end
    self.criterion = nn.MSECriterion()

    self.theta = {}
    self.theta.params, self.theta.grad_params = self.network:getParameters()

    collectgarbage()
    return self
end


function model:fp(params_, batch, sim)
    if params_ ~= self.theta.params then self.theta.params:copy(params_) end
    self.theta.grad_params:zero()  -- reset gradient

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
            this_past:size(2) == self.mp.input_dim)  -- TODO RESIZE THIS
    assert(context:size(1) == self.mp.batch_size and
            context:size(2)==self.mp.seq_length
            and context:size(3) == self.mp.input_dim)
    assert(this_future:size(1) == self.mp.batch_size and
            this_future:size(2) == self.mp.out_dim)  -- TODO RESIZE THIS

    -- here you have to create a table of tables
    -- this: (bsize, input_dim)
    -- context: (bsize, mp.seq_length, dim)
    local input = {}
    for t=1,torch.find(mask,1)[1] do  -- not actually mp.seq_length!
        table.insert(input, {this_past,torch.squeeze(context[{{},{t}}])})
    end

    local prediction = self.network:forward(input)

    local p_pos, p_vel, p_obj_prop =
                        unpack(split_output(self.mp):forward(prediction))
    local gt_pos, gt_vel, gt_obj_prop =
                        unpack(split_output(self.mp):forward(this_future))

    local loss = self.criterion:forward(p_vel, gt_vel)

    collectgarbage()
    return loss, prediction
end


-- local p_pos, p_vel, p_obj_prop=split_output(params):forward(prediction)
-- local gt_pos, gt_vel, gt_obj_prop=split_output(params):forward(this_future)
-- a lot of instantiations of split_output
function model:bp(batch, prediction, sim)
    self.theta.grad_params:zero() -- the d_parameters
    local this, context, y, mask = unpack(batch)
    local x = {this=this,context=context}

    if not sim then
        y = crop_future(y, {y:size(1), mp.winsize-mp.num_past, mp.object_dim},
                            {2,mp.num_future})
    end

    -- unpack inputs. All of these have been CUDAed already if need be
    local this_past     = convert_type(x.this:clone(), self.mp.cuda)
    local context       = convert_type(x.context:clone(), self.mp.cuda)
    local this_future   = convert_type(y:clone(), self.mp.cuda)


    local input = {}
    for t=1,torch.find(mask,1)[1] do  -- not actually mp.seq_length!
        table.insert(input, {this_past,context[{{},{t}}]})
    end

    local splitter = split_output(self.mp)

    local p_pos, p_vel, p_obj_prop = unpack(splitter:forward(prediction))
    local gt_pos, gt_vel, gt_obj_prop =
                        unpack(split_output(self.mp):forward(this_future))

    local d_pos = nn.IdentityCriterion():backward(p_pos, gt_pos)
    local d_vel = self.criterion:backward(p_vel, gt_vel)
    local d_obj_prop = nn.IdentityCriterion():backward(p_obj_prop, gt_obj_prop)
    local d_pred = splitter:backward({prediction}, {d_pos, d_vel, d_obj_prop})
    self.network:backward(input,d_pred)  -- updates grad_params
    collectgarbage()
    return self.theta.grad_params
end

return model
