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

    local world_state_pre, obj_prop_pre = split_tensor(3,
                {num_future, object_dim},{{1,4},{5,object_dim}})
                (decoder_preout):split(2)  -- contains info about objectdim!
    local obj_prop = nn.Sigmoid()(obj_prop_pre)
    local world_state = world_state_pre -- linear
    local dec_out_reshaped = nn.JoinTable(3)({world_state,obj_prop})
    local decoder_out = nn.Reshape(out_dim, true)(dec_out_reshaped)
    return nn.gModule({rnn_out}, {decoder_out})
end

function init_object_decoder_with_identity(rnn_hid_dim, num_future, object_dim)
    -- rnn_out had better be of dim (batch_size, rnn_hid_dim)
    local rnn_out = nn.Identity()()

    ------------------------------------------------
    -- input branch to decoder
    local orig_state = nn.Identity()()

    -- should I combine them first, or should I do a encoding then combine?
    -- I think I should just combine
    local decoder_in = nn.JoinTable(2)({orig_state, rnn_out})  -- TODO: figure out what dimension this is

    local decoder_preout, decoder_net
    if mp.decoder_layers == 0 then
        decoder_net = nn.Linear(decoder_in_dim, out_dim)
    else
        local decoder_net = nn.Sequential()
        for i=1,mp.num_layers do
            if i == 1 then 
                decoder_net:add(nn.Linear(decoder_in_dim, rnn_hid_dim))
                decoder_net:add(nn.ReLU())
            elseif i == num_layers then 
                decoder_net:add(nn.Linear(rnn_hid_dim, out_dim))
            else
                decoder_net:add(nn.Linear(rnn_hid_dim, rnn_hid_dim))
                decoder_net:add(nn.ReLU())
            end
            if mp.batch_norm then 
                decoder_net:add(nn.BatchNormalization(params.rnn_dim))
            end
        end
    end

    local out_dim = num_future * object_dim
    local decoder_preout = decoder_net(rnn_out)
    ------------------------------------------------

    local world_state_pre, obj_prop_pre = split_tensor(3,
                {num_future, object_dim},{{1,4},{5,object_dim}})
                (decoder_preout):split(2)  -- contains info about objectdim!
    local obj_prop = nn.Sigmoid()(obj_prop_pre)
    local world_state = world_state_pre -- linear
    local dec_out_reshaped = nn.JoinTable(3)({world_state,obj_prop})
    local decoder_out = nn.Reshape(out_dim, true)(dec_out_reshaped)
    return nn.gModule({orig_state, rnn_out}, {decoder_out})
end

-- with a bidirectional lstm, no need to put a mask
-- however, you can have variable sequence length now!
function init_network(params)
    -- encoder produces: (bsize, rnn_inp_dim)
    -- decoder expects (bsize, 2*rnn_hid_dim)

    local layer, sequencer_type, dcoef
    if params.model == 'lstmobj' then
        layer = nn.LSTM(params.rnn_dim,params.rnn_dim)
        sequencer_type = nn.BiSequencer
        dcoef = 2
    elseif params.model == 'gruobj' then
        layer = nn.GRU(params.rnn_dim,params.rnn_dim)
        sequencer_type = nn.BiSequencer
        dcoef = 2
    elseif params.model == 'ffobj' then
        layer = nn.Linear(params.rnn_dim, params.rnn_dim)
        sequencer_type = nn.Sequencer
        dcoef = 1
    else
        error('unknown model')
    end

    local encoder = init_object_encoder(params.input_dim, params.rnn_dim)
    local decoder = init_object_decoder(dcoef*params.rnn_dim, params.num_future,
                                                            params.object_dim)

    local step = nn.Sequential()
    step:add(encoder)
    for i = 1,params.layers do
        step:add(layer:clone())  -- same param initial, but weights not shared
        step:add(nn.ReLU())
        if mp.batch_norm then 
            step:add(nn.BatchNormalization(params.rnn_dim))
        end
    end

    local sequencer = sequencer_type(step)
    sequencer:remember('neither')

    -- I think if I add: sequencer_type(sequencer), then it'd be able to go through time as well.
    --
    local net = nn.Sequential()
    net:add(sequencer)

    -- input table of (bsize, 2*d_hid) of length seq_length
    -- output: tensor (bsize, 2*d_hid)
    net:add(nn.CAddTable())  -- add across the "timesteps" to sum contributions
    net:add(decoder)
    return net
end


function split_output(params)
    -- local POSVELDIM = 4
    local POSVELDIM = 6
    local future = nn.Identity()()

    local world_state, obj_prop = split_tensor(3,
        {params.num_future,params.object_dim},{{1,config_args.si.m[1]-1},{config_args.si.m[1],params.object_dim}})
        ({future}):split(2)

    -- split state: only pass gradients on velocity and angularVelocity
    local pos, vel, ang, ang_vel = split_tensor(3,
        {params.num_future, POSVELDIM},{{1,2},{3,4},{5,5},{6,6}})
        ({world_state}):split(4) -- split world_state in half on last dim

    local net = nn.gModule({future},{pos, vel, ang, ang_vel, obj_prop})
    if mp.cuda then net = net:cuda() end
    return net
end

-- boundaries: {{l1,r1},{l2,r2},{l3,r3},etc}
function split_tensor(dim, reshape, boundaries)
    local tensor = nn.Identity()()
    local reshaped = nn.Reshape(reshape[1],reshape[2], 1, true)(tensor)
    local splitted = nn.SplitTable(dim)(reshaped)
    local chunks = {}
    for cb = 1,#boundaries do
        local left,right = unpack(boundaries[cb])
        local length = right-left+1
        chunks[#chunks+1] = nn.JoinTable(dim)
                                (nn.NarrowTable(left,length)(splitted))
    end
    local net = nn.gModule({tensor},chunks)
    if mp.cuda then net = net:cuda() end
    return net
end


