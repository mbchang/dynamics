require 'nn'
require 'rnn'
require 'torch'
require 'nngraph'
require 'Base'
require 'IdentityCriterion'
require 'data_utils'
require 'modules'

nngraph.setDebug(true)

-- adapted from https://github.com/rahul-iisc/seq2seq-mapping/blob/master/seq2seq.lua


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

    local enclstm, declstm
    local encLSTMs = {}
    local decLSTMs = {}

    local allModContainer = nn.Container()

    assert(num_layers > 1)

    -- if mp.batch_norm then nn.FastLSTM.bn = true end  -- doesn't seem to do anything
    nn.FastLSTM.usenngraph = true

    -- Encoder
    -- input: table of length num_obj of (bsize, obj_dim)
    enclstm = nn.Sequential()

    local encoder_LSTM
    for i = 1, num_layers do
        nn.FastLSTM.usenngraph = true
        if i == 1 then encoder_LSTM = nn.FastLSTM(in_dim, hid_dim)
        else
            encoder_LSTM = nn.FastLSTM(hid_dim, hid_dim)
            if mp.dropout > 0 then enclstm:add(nn.Sequencer(nn.Dropout(mp.dropout))) end
        end
        enclstm:add(nn.Sequencer(encoder_LSTM))
        -- enclstm:add(nn.Sequencer(nn.ReLU()))  -- this should be fine
        allModContainer:add(encoder_LSTM)
        table.insert(encLSTMs, encoder_LSTM)
    end
    enclstm:add(nn.SelectTable(-1))  -- get last timestep


    -- Decoder
    -- should the first layer be (hid_dim, hid_dim) or (2*in_dim, hid_dim)? 
    -- basically I think the first layer of the decoder is supposed to take the output of the decoder as input
    -- so whatever dimension is the decoder output, that will be the decoder input
    -- note that the decoder shares completely different weights from the encoder
    declstm = nn.Sequential()

    local decoder_LSTM
    for i = 1, num_layers do
        nn.FastLSTM.usenngraph = true
        if i == 1 then decoder_LSTM = nn.FastLSTM(out_dim, hid_dim);  -- the first LSTM in decoder LSTM stack. 
        else
            decoder_LSTM = nn.FastLSTM(hid_dim, hid_dim)
            if mp.dropout > 0 then declstm:add(nn.Sequencer(nn.Dropout(mp.dropout))) end
        end
        declstm:add(nn.Sequencer(decoder_LSTM))
        -- declstm:add(nn.Sequencer(nn.ReLU()))  -- this should be fine
        allModContainer:add(decoder_LSTM)
        table.insert(decLSTMs, decoder_LSTM)
    end
    
    local to_output = nn.Linear(hid_dim, out_dim)
    declstm:add(nn.Sequencer(to_output))
    allModContainer:add(to_output)

    -- so now we have enclstm and declstm and allModContainer
    -- don't need to call remember because each training example is an independent sequence
    return enclstm, declstm, allModContainer, encLSTMs, decLSTMs
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
        -- TODO!
        local checkpoint = torch.load(model_path)
        self.enclstm = checkpoint.model.enclstm:clone()
        self.declstm = checkpoint.model.declstm:clone()
        self.encLSTMs = checkpoint.model.encLSTMs
        self.decLSTMs = checkoint.model.decLSTMs
        self.network = checkpoint.model.network:clone()  -- TODO VERIFY THIS!
        self.criterion = checkpoint.model.criterion:clone()
        self.identitycriterion = checkpoint.model.identitycriterion:clone()
        if self.mp.cuda then self:cuda() end
    else
        self.criterion = nn.MSECriterion(false)  -- not size averaging!
        self.identitycriterion = nn.IdentityCriterion()
        self.enclstm, self.declstm, self.network, self.encLSTMs, self.decLSTMs = init_network(self.mp)
        if self.mp.cuda then self:cuda() end
    end

    self.theta = {}
    self.theta.params, self.theta.grad_params = self.network:getParameters()

    print('Encoder')
    print(self.enclstm)
    print('Decoder')
    print(self.declstm)
    collectgarbage()
    return self
end

function model:cuda()
    -- self.enclstm:cuda()
    -- self.declstm:cuda()
    self.network = self.network:cuda()
    self.criterion:cuda()
    self.identitycriterion:cuda()
end

function model:float()
    -- self.enclstm:float()
    -- self.declstm:float()
    self.network = self.network:float()
    self.criterion:float()
    self.identitycriterion:float()
end

function model:clearState()
    self.enclstm:clearState()
    self.declstm:clearState()
end

-- helper functions to for encoder-decoder coupling.
--[[ Forward coupling: Copy encoder cell and output to decoder LSTM ]]--
function model:forwardConnect(encLSTMs, decLSTMs)
    seqLen = #(encLSTMs[1].outputs)
    for i = 1, #encLSTMs do
        local encLSTM, decLSTM = encLSTMs[i], decLSTMs[i]
        decLSTM.userPrevOutput = nn.rnn.recursiveCopy(decLSTM.userPrevOutput, encLSTM.outputs[seqLen])
        decLSTM.userPrevCell = nn.rnn.recursiveCopy(decLSTM.userPrevCell, encLSTM.cells[seqLen])
    end
end

--[[ Backward coupling: Copy decoder gradients to encoder LSTM ]]--
function model:backwardConnect(encLSTMs, decLSTMs)
    for i = 1, #encLSTMs do
        local encLSTM, decLSTM = encLSTMs[i], decLSTMs[i]
        encLSTM.userNextGradCell = nn.rnn.recursiveCopy(encLSTM.userNextGradCell, decLSTM.userGradPrevCell)
        encLSTM.gradPrevOutput = nn.rnn.recursiveCopy(encLSTM.gradPrevOutput, decLSTM.userGradPrevOutput)
    end
end

-- you need to put in padding here
-- encIn: table of length (num_obj) of (bsize, in_dim).
-- decIn: table of length (num_obj+1) of (bsize, out_dim). First elemnt is <GO>
-- decTarget: table of length (num_obj+1) of (bsize, out_dim). Last elemtn is <EOS>
function model:unpack_batch(batch, sim)
    local this, context, this_future, context_future, mask = unpack(batch)
    local past = convert_type(torch.cat({unsqueeze(this:clone(),2), context},2), mp.cuda)
    local future = convert_type(torch.cat({unsqueeze(this_future:clone(),2), context_future},2), mp.cuda)

    local bsize, num_obj = past:size(1), past:size(2)
    local num_past, num_future = past:size(3), future:size(3)
    local obj_dim = past:size(4)

    -- now break into different trajectories
    local encIn = {}  -- seq_length: the first element has a symbol [10], the last has a symbol [01]
    local decIn = {}  -- seq_length: the first element has a symbol [10]
    local decTarget = {}  -- seq_length: the last element has a symbol [01]

    -- shuffle
    local shuffind = torch.randperm(num_obj)

    local start_symbol = torch.zeros(bsize,2)
    start_symbol[{{},{1}}]:fill(1)  -- [10]
    start_symbol = convert_type(start_symbol, mp.cuda)

    local stop_symbol = torch.zeros(bsize,2)
    stop_symbol[{{},{2}}]:fill()  -- [01]
    stop_symbol = convert_type(start_symbol, mp.cuda)  

    local continue_symbol = torch.zeros[(bsize,2)
    continue_symbol = convert_type(start_symbol, mp.cuda)

    -- encoder
    for i = 1,num_obj do
        if i == 1 then
            encIn[i] = torch.cat({past[{{},{shuffind[i]}}]:reshape(bsize,num_past*obj_dim),start_symbol},2)
        elseif i == num_obj then
            encIn[i] = torch.cat({past[{{},{shuffind[i]}}]:reshape(bsize,num_past*obj_dim),stop_symbol},2)
        else
            encIn[i] = torch.cat({past[{{},{shuffind[i]}}]:reshape(bsize,num_past*obj_dim),continue_symbol},2)
        end
    end

    -- decoder
    -- but how would you represent the start symbol in the decoder? I could give it a matrix of zeros?
    for i =1,num_obj do
        if i == 1 then
            decIn[i] = torch.cat({future[{{},{shuffind[i]}}]:reshape(bsize,num_future*obj_dim),start_symbol},2)
            decTarget[i] = future[{{},{shuffind[i]}}]:reshape(bsize,num_future*obj_dim)
        elseif i == num_obj then

            encIn[i] = torch.cat({past[{{},{shuffind[i]}}]:reshape(bsize,num_past*obj_dim),stop_symbol},2)
            decIn[i] = torch.cat({future[{{},{shuffind[i]}}]:reshape(bsize,num_future*obj_dim),stop_symbol},2)   
        else

            encIn[i] = torch.cat({past[{{},{shuffind[i]}}]:reshape(bsize,num_past*obj_dim),continue_symbol},2)
            decIn[i] = torch.cat({future[{{},{shuffind[i]}}]:reshape(bsize,num_future*obj_dim),continue_symbol},2) 


        table.insert(encIn,past[{{},{shuffind[i]}}]:reshape(bsize,num_past*obj_dim))
        decIn[i+1] = future[{{},{shuffind[i]}}]:reshape(bsize,num_future*obj_dim)
        decTarget[i] = future[{{},{shuffind[i]}}]:reshape(bsize,num_future*obj_dim)
    end

    -- works
    return encIn, decIn, decTarget
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

    -- print(self.theta.params:norm())

    local encIn, decIn, decTarget = self:unpack_batch(batch, sim)

    --  forward pass
    local encOut = self.enclstm:forward(encIn)
    self:forwardConnect(self.encLSTMs, self.decLSTMs)
    local prediction = self.declstm:forward(decIn)  -- table of length (num_obj + 1)


    -- do I want to predict the <EOS> too? --> NO
    -- the loss won't be computed based on EOS

    local loss = 0
    for i = 1,#prediction-1 do
        -- table of length num_obj of {bsize, num_future, obj_dim}
        local p_pos, p_vel, p_ang, p_ang_vel, p_obj_prop =
                            unpack(split_output(self.mp):forward(prediction[i]))
        local gt_pos, gt_vel, gt_ang, gt_ang_vel, gt_obj_prop =
                            unpack(split_output(self.mp):forward(decTarget[i]))

        local loss_vel = self.criterion:forward(p_vel, gt_vel)
        local loss_ang_vel = self.criterion:forward(p_ang_vel, gt_ang_vel)
        local obj_loss = loss_vel + loss_ang_vel
        obj_loss = obj_loss/(p_vel:nElement()+p_ang_vel:nElement()) -- manually do size average
        loss = loss + obj_loss
    end
    loss = loss/(#prediction-1)

    collectgarbage()
    return loss, prediction
end

-- encIn: table of length (num_obj) of (bsize, in_dim).
-- decIn: table of length (num_obj+1) of (bsize, out_dim). First elemnt is <GO>
-- decTarget: table of length (num_obj+1) of (bsize, out_dim). Last elemtn is <EOS>


-- local p_pos, p_vel, p_obj_prop=split_output(params):forward(prediction)
-- local gt_pos, gt_vel, gt_obj_prop=split_output(params):forward(this_future)
-- a lot of instantiations of split_output
function model:bp(batch, prediction, sim)
    self.theta.grad_params:zero() -- the d_parameters
    local encIn, decIn, decTarget = self:unpack_batch(batch, sim)

    local splitter = split_output(self.mp)

    -- you will still backpropagate from EOS

    local d_pred = {}
    for i = 1, #prediction do  -- EOS is just all 1s in the output

        local p_pos, p_vel, p_ang, p_ang_vel, p_obj_prop = unpack(splitter:forward(prediction[i]))
        local gt_pos, gt_vel, gt_ang, gt_ang_vel, gt_obj_prop =
                            unpack(split_output(self.mp):forward(decTarget[i]))

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

        local obj_d_pred = splitter:backward({prediction[i]}, {d_pos, d_vel, d_ang, d_ang_vel, d_obj_prop}):clone()

        table.insert(d_pred, obj_d_pred)
    end

    -- still backpropagate from EOS
    -- self.criterion:forward(prediction[#prediction], decTarget[#prediction])
    -- local d_eos = self.criterion:backward(prediction[#prediction], decTarget[#prediction]):clone()
    -- table.insert(d_pred, d_eos)

    self.declstm:backward(decIn, d_pred)
    self:backwardConnect(self.encLSTMs, self.decLSTMs)
    -- zero gradient into encoder because we already did backward Connect
    self.enclstm:backward(encIn, convert_type(torch.zeros(mp.batch_size, mp.rnn_dim), mp.cuda))

    collectgarbage()
    return self.theta.grad_params
end

function model:sim(batch, numsteps)

end

return model
