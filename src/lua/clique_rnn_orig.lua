require 'nn'
require 'nngraph'
require 'rnn'
require 'modules'

config_args = require 'config'
params = {model='crnn', rnn_dim=100, input_dim=10, num_future=1, object_dim=10, layers=3, num_obj=4}
mp = {cuda=false}


function init_tstep_one_obj(params)
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
    elseif params.model == 'crnn' then
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
    local tstep_one_obj = nn.Sequential()
    tstep_one_obj:add(sequencer)


    -- input table of (bsize, 2*d_hid) of length seq_length
    -- output: tensor (bsize, 2*d_hid)
    tstep_one_obj:add(nn.CAddTable())  -- add across the "timesteps" to sum contributions
    tstep_one_obj:add(decoder)

    return tstep_one_obj
end

-- for sequencer: input table of (bsize, 2*d_hid) of length seq_length
-- for clique_rnn: input table of {(bsize, 2*d_hid) of length seq_length} of length timesteps


-- input: table of length num_obj of {table of length num_obj-1 of pairs of (bsize, input_dim)}
-- output: table of length num_obj of {(bsize, input_dim)}
function init_tstep(params)
    local tstep = nn.Sequencer(init_tstep_one_obj(params))
    return tstep
end

local timesteps = 5
local num_obj = 4
local bsize = 2



-- 2) pairwise_regroup
-- input: obj_states: table of length num_obj of {(bsize, input_dim)}
-- output: obj_state_pairs: table of length num_obj of {table of length num_obj-1 of pairs of (bsize, input_dim)}
function init_pairwise_regroup(params)
    local obj_states = nn.Identity()()
    local obj_state_pairs = {}
    for i=1,params.num_obj do
        local focus = nn.SelectTable(i)(obj_states)
        local pairs_for_focus = {}
        for j=1,params.num_obj do  -- how to deal with this? If I can deal with this then I'm good
            local context = nn.SelectTable(j)(obj_states)
            if i ~= j then table.insert(pairs_for_focus, nn.Identity(){focus, context}) end
        end
        table.insert(obj_state_pairs, nn.Identity()(pairs_for_focus))
    end

    local pairwise_regroup = nn.gModule({obj_states},obj_state_pairs)
    return pairwise_regroup
end

--------------------------------------------------------------------------------
-- input: table of length num_obj of {table of length num_obj-1 of pairs of (bsize, input_dim)}
-- output: table of length num_obj of {table of length num_obj-1 of pairs of (bsize, input_dim)}
function init_clique_rnn(params)
    local net = nn.Sequential()
    net:add(init_tstep(params))
    net:add(init_pairwise_regroup(params))
    -- here do I add memory, or do I add memory in tstep? Check how LSTM is being implemented
    -- perhaps I should look inthe documentation of AbstractRecurrent for SharedClones
    -- the tstep_one_obj should be cloned for all objects, but given an object, it should also persist through time.

    --------------------------------------------------------------------------------


    -- This is my network! But where do I store the internal memory?

    -- input: table of length timesteps of {table of length num_obj of {table of length num_obj-1 of pairs of (bsize, input_dim)}}
    -- output: table of length timesteps of {table of length num_obj of {table of length num_obj-1 of pairs of (bsize, input_dim)}}
    local clique_rnn = nn.Sequencer(net)
    -- print(clique_rnn:forward(input))
    return clique_rnn
end

-- LSTM internal state for each object
function init_clique_rnn_memory(params)
    local memory_net = nn.Sequential()
    memory_net:add(init_tstep(params))

    -- here create a table of LSTMS  --> how does this work with Sequential()
    local memory_in = nn.Identity()()  -- Parallel table?
    local memory_lstm = nn.Sequential()
    memory_lstm:add(nn.Linear(params.input_dim, params.rnn_dim))
    memory_lstm:add(nn.LSTM(params.rnn_dim,params.rnn_dim))
    memory_lstm:add(nn.Linear(params.rnn_dim, params.input_dim))
    -- local memory_cell = nn.ParallelTable()
    -- for i=1,params.num_obj do  -- TODO! WHAT TO DO ABOUT THIS? Should I use a Sequencer?
    --     memory_cell:add(memory_lstm)
    -- end

    local memory_cell = nn.Sequencer(memory_lstm)  -- goes through the object space
    -- TODO: what type of forgetting behavior do we want?

    local memory_out = memory_cell(memory_in)
    local memory_module = nn.gModule({memory_in}, {memory_out})

    memory_net:add(memory_module)
    memory_net:add(init_pairwise_regroup(params))

    local clique_rnn_memory = nn.Sequencer(memory_net)
    return clique_rnn_memory
end

--------------------------------------------------------------------------------
-- networks
--------------------------------------------------------------------------------
local tstep_one_obj = init_tstep_one_obj(params)
local tstep = init_tstep(params)
local clique_rnn = init_clique_rnn(params)
local clique_rnn_memory = init_clique_rnn_memory(params)


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
        -- self.network = nn.Linear(2,2)--init_clique_rnn_memory(self.mp)
        self.tstep = init_tstep(self.mp) 

        if self.mp.cuda then
            self.network:cuda()
            self.criterion:cuda()
            self.identitycriterion:cuda()
        end
    end

    self.theta = {}
    self.theta.params, self.theta.grad_params = self.tstep:getParameters()

    collectgarbage()
    return self
end

function model:unpack_batch(batch, sim)
    -- for the model
    -- input: table of length num_obj of {table of length num_obj-1 of pairs of (bsize, input_dim)}
    -- output: table of length num_obj of {table of length num_obj-1 of pairs of (bsize, input_dim)}


    -- past is reshaped, future is not
    local this, context, this_future, context_future, mask = unpack(batch)
    local past = torch.cat({unsqueeze(this:clone(),2), context},2)   -- (bsize, num_obj, num_past, obj_dim)
    local future = torch.cat({unsqueeze(this_future:clone(),2), context_future},2)  -- (bsize, num_obj, num_future, obj_dim)
    local trajectories = torch.cat({past, future},3)  -- (bsize, num_obj, winsize, obj_dim)

    local max_obj = self.mp.seq_length -- TODO: HARDCODED
    local bsize, num_obj = past:size(1), past:size(2)
    local num_past, num_future = past:size(3), future:size(3)
    local obj_dim = past:size(4)
    local timesteps = num_past + num_future

    -- here construct the input
    local input = {}  -- predict for all timesteps
    local target = {} -- this depends on how many future timesteps you are predicting and whether you want to do encoder decoder or something
    for t=1,timesteps do
        local tstep_input = {}
        for i=1,num_obj do
            local obj_input = {}  -- predict for this particular object
            local focus = trajectories[{{},{i},{t}}]
            for j=1,num_obj do
                if not(i==j) then 
                    local context = trajectories[{{},{j},{t}}]
                    table.insert(obj_input, {focus, context})
                end
            end
            table.insert(tstep_input, obj_input)
        end
        if t <= num_past then 
            table.insert(input, tstep_input)
        else
            table.insert(target, tstep_input)
        end
    end

    return input, target
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

    local input, target = unpack_batch(batch, sim)


    print('input')
    print(input)
    local effects = self.tstep(input)
    print('effects')
    print(effects)

    assert(false)

    local prediction = self.network:forward(input)

    local p_pos, p_vel, p_ang, p_ang_vel, p_obj_prop =
                        unpack(split_output(self.mp):forward(prediction))
    local gt_pos, gt_vel, gt_ang, gt_ang_vel, gt_obj_prop =
                        unpack(split_output(self.mp):forward(this_future))

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
function model:bp(batch, prediction, sim)
    self.theta.grad_params:zero() -- the d_parameters
    local input, this_future = unpack_batch(batch, sim)

    local splitter = split_output(self.mp)

    local p_pos, p_vel, p_ang, p_ang_vel, p_obj_prop = unpack(splitter:forward(prediction))
    local gt_pos, gt_vel, gt_ang, gt_ang_vel, gt_obj_prop =
                        unpack(split_output(self.mp):forward(this_future))

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

    local d_pred = splitter:backward({prediction}, {d_pos, d_vel, d_ang, d_ang_vel, d_obj_prop})
    self.network:backward(input,d_pred)  -- updates grad_params

    collectgarbage()
    return self.theta.grad_params
end

-- return model


-- -- --------------------------------------------------------------------------------
-- -- -- Data
-- -- --------------------------------------------------------------------------------

-- -- for one object
-- local tstep_one_obj_input = {}
-- for o=1,num_obj-1 do
--     table.insert(tstep_one_obj_input, {torch.rand(bsize, params.input_dim), torch.rand(bsize, params.input_dim)})
-- end

-- -- input: table of length num_obj of pairs of (bsize, input_dim)
-- -- output: (bsize, input_dim)
-- print(tstep_one_obj)
-- print(tstep_one_obj:forward(tstep_one_obj_input))
-- --------------------------------------------------------------------------------


-- -- for all objects
local tstep_input = {}
for n=1,num_obj do
    local obj_input = {}  -- predict for this particular object
    for o=1,num_obj-1 do
        table.insert(obj_input, {torch.rand(bsize, params.input_dim), torch.rand(bsize, params.input_dim)})
    end
    table.insert(tstep_input, obj_input)
end


print(tstep)
local tstep_output = tstep:forward(tstep_input)
print(tstep_output)

-- --------------------------------------------------------------------------------

-- -- for all timesteps
-- local input = {}  -- predict for all timesteps
-- for t=1,timesteps do
--     local tstep_input = {}
--     for n=1,num_obj do
--         local obj_input = {}  -- predict for this particular object
--         for o=1,num_obj-1 do
--             table.insert(obj_input, {torch.rand(bsize, params.input_dim), torch.rand(bsize, params.input_dim)})
--         end
--         table.insert(tstep_input, obj_input)
--     end
--     table.insert(input, tstep_input)
-- end

-- -- so basically the recurrence is: tstep --> pairwise_regroup --> tstep --> pairwise_regroup etc
-- --------------------------------------------------------------------------------

-- print(clique_rnn_memory:forward(input))
