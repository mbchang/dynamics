require 'nn'
require 'rnn'
require 'torch'
require 'nngraph'
require 'IdentityCriterion'
require 'data_utils'

nngraph.setDebug(true)

function init_object_encoder(input_dim, rnn_inp_dim, bias)
    assert(rnn_inp_dim % 2 == 0)
    local thisp     = nn.Identity()() -- (batch_size, input_dim)
    local contextp  = nn.Identity()() -- (batch_size, partilce_dim)

    -- (batch_size, rnn_inp_dim/2)
    local thisp_out     = nn.ReLU()
                            (nn.Linear(input_dim, rnn_inp_dim/2, bias)(thisp))
    local contextp_out  = nn.ReLU()
                            (nn.Linear(input_dim, rnn_inp_dim/2, bias)(contextp))

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
                {num_future, object_dim},{{1,6},{7,object_dim}})
                (decoder_preout):split(2)  -- contains info about objectdim!
    local obj_prop = nn.Sigmoid()(obj_prop_pre)
    local world_state = world_state_pre -- linear
    local dec_out_reshaped = nn.JoinTable(3)({world_state,obj_prop})
    local decoder_out = nn.Reshape(out_dim, true)(dec_out_reshaped)
    return nn.gModule({rnn_out}, {decoder_out})
end

function init_object_decoder_with_identity(rnn_hid_dim, num_layers, num_past, num_future, object_dim, identity_dim)
    -- rnn_out (batch_size, rnn_hid_dim)
    local rnn_out = nn.Identity()()
    local out_dim = num_future * object_dim

    ------------------------------------------------
    -- input branch to decoder
    -- orig_state (batch_size, mp.num_past*mp.object_dim)
    local orig_state = nn.Identity()()
    local decoder_in_dim = identity_dim + rnn_hid_dim

    -- should I combine them first, or should I do a encoding then combine?
    -- I think I should just combine
    local decoder_in = nn.JoinTable(2)({rnn_out, orig_state})

    local decoder_preout, decoder_net
    if num_layers == 0 or num_layers == 1 then
        decoder_net = nn.Linear(decoder_in_dim, out_dim)
    else
        decoder_net = nn.Sequential()
        for i=1,num_layers do
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

    local decoder_preout = decoder_net(decoder_in)

    local world_state_pre, obj_prop_pre = split_tensor(3,
                {num_future, object_dim},{{1,6},{7,object_dim}})
                (decoder_preout):split(2)  -- contains info about objectdim!
    local obj_prop = nn.Sigmoid()(obj_prop_pre)
    local world_state = world_state_pre -- linear
    local dec_out_reshaped = nn.JoinTable(3)({world_state,obj_prop})
    local decoder_out = nn.Reshape(out_dim, true)(dec_out_reshaped)
    return nn.gModule({rnn_out, orig_state}, {decoder_out})
end


function split_output(params)
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
    if mp.cuda then net:cuda() end
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
    if mp.cuda then net:cuda() end
    return net
end


