require 'nn'
require 'nngraph'
require 'rnn'

config_args = require 'config'
params = {model='ffobj', rnn_dim=100, input_dim=10, num_future=1, object_dim=10, layers=3}
mp = {cuda=false}


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

    if mp.cuda then
        return net:cuda()
    else
        return net
    end
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
    if mp.cuda then
        return net:cuda()
    else
        return net
    end
end

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
    end

    local sequencer = sequencer_type(step)
    sequencer:remember('neither')

    -- I think if I add: sequencer_type(sequencer), then it'd be able to go through time as well.
    --
    local tstep = nn.Sequential()
    tstep:add(sequencer)


    -- input table of (bsize, 2*d_hid) of length seq_length
    -- output: tensor (bsize, 2*d_hid)
    tstep:add(nn.CAddTable())  -- add across the "timesteps" to sum contributions
    tstep:add(decoder)

    return tstep
end

-- for sequencer: input table of (bsize, 2*d_hid) of length seq_length
-- for clique_rnn: input table of {(bsize, 2*d_hid) of length seq_length} of length timesteps


local timesteps = 5
local num_obj = 4
local bsize = 2

-- for one object
local tstep_one_obj_input = {}
for o=1,num_obj-1 do
    table.insert(tstep_one_obj_input, {torch.rand(bsize, params.input_dim), torch.rand(bsize, params.input_dim)})
end

-- input: table of length num_obj of pairs of (bsize, input_dim)
-- output: (bsize, input_dim)
local tstep_one_obj = init_network(params)
print(tstep_one_obj)
print(tstep_one_obj:forward(tstep_one_obj_input))
--------------------------------------------------------------------------------


-- for all objects
local tstep_input = {}
for n=1,num_obj do
    local obj_input = {}  -- predict for this particular object
    for o=1,num_obj-1 do
        table.insert(obj_input, {torch.rand(bsize, params.input_dim), torch.rand(bsize, params.input_dim)})
    end
    table.insert(tstep_input, obj_input)
end

-- input: table of length num_obj of {table of length num_obj-1 of pairs of (bsize, input_dim)}
-- output: table of length num_obj of {(bsize, input_dim)}
local tstep = nn.Sequencer(tstep_one_obj)
print(tstep)
local tstep_output = tstep:forward(tstep_input)
print(tstep_output)

--------------------------------------------------------------------------------

-- for all timesteps
local input = {}  -- predict for all timesteps
for t=1,timesteps do
    local tstep_input = {}
    for n=1,num_obj do
        local obj_input = {}  -- predict for this particular object
        for o=1,num_obj-1 do
            table.insert(obj_input, {torch.rand(bsize, params.input_dim), torch.rand(bsize, params.input_dim)})
        end
        table.insert(tstep_input, obj_input)
    end
    table.insert(input, tstep_input)
end

-- so basically the recurrence is: tstep --> pairwise_regroup --> tstep --> pairwise_regroup etc

-- need a sequential here
-- 1) tstep
-- 2) pairwise_regroup


-- 2) pairwise_regroup
-- input: obj_states: table of length num_obj of {(bsize, input_dim)}
-- output: obj_state_pairs: table of length num_obj of {table of length num_obj-1 of pairs of (bsize, input_dim)}

local obj_states = nn.Identity()()
local obj_state_pairs = {}
for i=1,num_obj do
    local focus = nn.SelectTable(i)(obj_states)
    local pairs_for_focus = {}
    for j=1,num_obj do
        local context = nn.SelectTable(j)(obj_states)
        if i ~= j then table.insert(pairs_for_focus, nn.Identity(){focus, context}) end
    end
    table.insert(obj_state_pairs, nn.Identity()(pairs_for_focus))
end

local pairwise_regroup = nn.gModule({obj_states},obj_state_pairs)

--------------------------------------------------------------------------------
-- input: table of length num_obj of {table of length num_obj-1 of pairs of (bsize, input_dim)}
-- output: table of length num_obj of {table of length num_obj-1 of pairs of (bsize, input_dim)}
local net = nn.Sequential()
net:add(tstep)
net:add(pairwise_regroup)
-- here do I add memory, or do I add memory in tstep? Check how LSTM is being implemented
-- perhaps I should look inthe documentation of AbstractRecurrent for SharedClones
-- the tstep_one_obj should be cloned for all objects, but given an object, it should also persist through time.

--------------------------------------------------------------------------------


-- This is my network! But where do I store the internal memory?

-- input: table of length timesteps of {table of length num_obj of {table of length num_obj-1 of pairs of (bsize, input_dim)}}
-- output: table of length timesteps of {table of length num_obj of {table of length num_obj-1 of pairs of (bsize, input_dim)}}
local clique_rnn = nn.Sequencer(net)
-- print(clique_rnn:forward(input))


--------------------------------------------------------------------------------
-- LSTM internal state for each object
local memory_net = nn.Sequential()
memory_net:add(tstep)

-- here create a table of LSTMS  --> how does this work with Sequential()
local memory_in = nn.Identity()()  -- Parallel table?
local memory_lstm = nn.Sequential()
memory_lstm:add(nn.Linear(params.input_dim, params.rnn_dim))
memory_lstm:add(nn.LSTM(params.rnn_dim,params.rnn_dim))
memory_lstm:add(nn.Linear(params.rnn_dim, params.input_dim))
local memory_cell = nn.ParallelTable()
for i=1,num_obj do
    memory_cell:add(memory_lstm)
end
local memory_out = memory_cell(memory_in)
local memory_module = nn.gModule({memory_in}, {memory_out})

memory_net:add(memory_module)
memory_net:add(pairwise_regroup)

-- print('assdfdfdfdfdfd')
-- print(memory_module:forward(tstep_output))

local clique_rnn_memory = nn.Sequencer(net)
print(clique_rnn_memory:forward(input))
