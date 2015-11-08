-- Train DDCIGN 

require 'metaparams'
require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'model'
require 'image'

if common_mp.cuda then
    require 'cutorch'
    require 'cunn'
end

if common_mp.cudnn then
    require 'cudnn'
end

local DataLoader = require 'DataLoader'
local model_utils = require 'model_utils'


local Tester = {}
Tester.__index = Tester

function Tester.create(dataset, mp)
    local self = {}
    setmetatable(self, Tester)
    self.mp = mp
    self.dataset = dataset  -- string for the dataset folder
    self.test_loader = DataLoader.create(self.dataset, self.mp.seq_length, self.mp.batch_size, self.mp.shuffle)
    collectgarbage()
    return self
end


function Tester:load_model(model)
    ------------------------------------ Create Model ------------------------------------
    self.protos                 = model -- torch.load(modelfile)
    self.protos.criterion       = nn.MSECriterion()

    if common_mp.cuda then 
        self.protos.encoder:cuda()
        self.protos.lstm:cuda()
        self.protos.decoder:cuda()
        self.protos.criterion:cuda()
    end

    ------------------------------------- Parameters -------------------------------------
    self.theta = {}
    self.theta.params, self.theta.grad_params = model_utils.combine_all_parameters(self.protos.encoder, self.protos.lstm, self.protos.decoder) 

    ------------------------------------ Clone Model -------------------------------------
    self.clones = {}
    for name,proto in pairs(self.protos) do
        self.clones[name] = model_utils.clone_many_times(proto, self.mp.seq_length, not proto.parameters)  -- clone 1 times
    end

    -------------------------------- Initialize LSTM State -------------------------------
    self.lstm_init_state = {}
    -- LSTM initial state (zero initially, but final state gets sent to initial state when we do BPTT)
    self.lstm_init_state.initstate_c = torch.zeros(self.mp.batch_size, self.mp.LSTM_hidden_dim)
    self.lstm_init_state.initstate_h = self.lstm_init_state.initstate_c:clone()

    if common_mp.cuda then
        self.lstm_init_state.initstate_c = self.lstm_init_state.initstate_c:cuda()
        self.lstm_init_state.initstate_h = self.lstm_init_state.initstate_h:cuda()
    end
    collectgarbage()
end


function Tester:forward_pass_test(params_, x, y)
    if params_ ~= self.theta.params then
        self.theta.params:copy(params_)
    end
    self.theta.grad_params:zero()  -- reset gradient

    ------------------ get minibatch -------------------
    -- local x, y = self.test_loader:next_batch(self.mp)  -- the way it is defined in loader is to just keep cycling through the same dataset
    local test_loss = 0

    ------------------- forward pass -------------------
    local embeddings = {}
    local lstm_c = {[0]=self.lstm_init_state.initstate_c} -- internal cell states of LSTM
    local lstm_h = {[0]=self.lstm_init_state.initstate_h} -- output values of LSTM
    local predictions = {}

    for t = 1,self.mp.seq_length do
        embeddings[t] = self.clones.encoder[t]:forward(torch.squeeze(x[{{},{t}}]))
        lstm_c[t], lstm_h[t] = unpack(self.clones.lstm[t]:forward{embeddings[t], lstm_c[t-1], lstm_h[t-1]})
        predictions[t] = self.clones.decoder[t]:forward(lstm_h[t])
        test_loss = test_loss + self.clones.criterion[t]:forward(predictions[t], torch.squeeze(y[{{},{t}}]))

        -- DEBUGGING
        -- image.display(predictions[t])  -- here you can also save the image
    end
    collectgarbage()
    return test_loss
end


function Tester:test(modelfile, num_iters)
    self:load_model(modelfile)
    local sum_loss = 0
    for i = 1, num_iters do 
        local x, y = self.test_loader:next_batch(self.mp)
        local test_loss = self:forward_pass_test(self.theta.params, x, y) 
        sum_loss = sum_loss + test_loss
    end
    local avg_loss = sum_loss/num_iters
    collectgarbage()
    return avg_loss
end

return Tester

