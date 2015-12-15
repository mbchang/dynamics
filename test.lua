-- Train 

require 'metaparams'
require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'model'
require 'image'
require 'torchx'
require 'hdf5'
require 'data_utils'

if common_mp.cuda then require 'cutorch' end
if common_mp.cunn then require 'cunn' end

local DataLoader = require 'DataLoader'
local model_utils = require 'model_utils'


local Tester = {}
Tester.__index = Tester

function Tester.create(dataset, mp)
    local self = {}
    setmetatable(self, Tester)
    self.mp = mp
    self.dataset = dataset  -- string for the dataset folder
    self.test_loader = DataLoader.create(self.dataset, self.mp.dataset_folder, self.mp.batch_size, self.mp.curriculum, self.mp.shuffle)
    collectgarbage()
    return self
end


function Tester:load_model(model)
    ------------------------------------ Create Model ------------------------------------
    self.network                 = model -- torch.load(modelfile)
    if common_mp.cuda then self.network:cuda() end

    ------------------------------------- Parameters -------------------------------------
    -- self.theta = {}
    -- self.theta.params, self.theta.grad_params = self.network:getParameters()

    ------------------------------------ Clone Model -------------------------------------
    -- self.rnns = g_cloneManyTimes(self.network, self.mp.seq_length, not self.network.parameters)
    self.rnns = model_utils.clone_many_times(self.network, self.mp.seq_length, not self.network.parameters)

    -- This will cache the values that s takes on in one forward pass
    self.s = {}
    for j = 0, self.mp.seq_length do
        self.s[j] = {}
        for d = 1, 2 * self.mp.layers do
            self.s[j][d] = model_utils.transfer_data(torch.zeros(self.mp.batch_size, self.mp.rnn_dim), common_mp.cuda) 
        end
    end
    collectgarbage()
end

function Tester:reset_state()
    for j = 0, self.mp.seq_length do
        for d = 1, 2 * self.mp.layers do
            self.s[j][d]:zero()
        end
    end
end


function Tester:forward_pass_test(params_, x, y)
    -- if params_ ~= self.theta.params then self.theta.params:copy(params_) end
    -- self.theta.grad_params:zero()  -- reset gradient
    self:reset_state()  -- reset s

    ------------------ get minibatch -------------------
    local this_past     = model_utils.transfer_data(x.this:clone(), common_mp.cuda)
    local context       = model_utils.transfer_data(x.context:clone(), common_mp.cuda)
    local this_future   = model_utils.transfer_data(y:clone(), common_mp.cuda)

    ------------------- forward pass -------------------
    local loss = model_utils.transfer_data(torch.zeros(self.mp.seq_length), common_mp.cuda)
    local predictions = {}
    for i = 1, self.mp.seq_length do
        local sim1 = self.s[i-1]  -- had been reset to 0 for initial pass
        loss[i], self.s[i], predictions[i] = unpack(self.rnns[i]:forward({this_past, context[{{},i}], sim1, this_future}))  -- problem! (feeding thisp_future every time; is that okay because I just update the gradient based on desired timesstep?)
    end 

    collectgarbage()
    return loss:sum(), predictions  -- we sum the loss through time
end


function Tester:test(model, params_, num_iters)
    self:load_model(model)
    local sum_loss = 0
    for i = 1, num_iters do 
        local this, context, y, mask = unpack(self.test_loader:next_batch())  -- the way it is defined in loader is to just keep cycling through the same dataset
        local test_loss, all_preds = self:forward_pass_test(params_, {this=this,context=context}, y)
        sum_loss = sum_loss + test_loss

        -- here you have the option to save predictions into a file
        local prediction = all_preds[torch.find(mask,1)[1]] -- this would be your 'this', or you could shift over, or do other interesting things
        -- For now, just save it as hdf5. You can feed it back in later if you'd like
        save_to_hdf5('my_pred.h5', 'pred', prediction)  -- works, but you should change the name
    end
    local avg_loss = sum_loss/num_iters  
    collectgarbage()
    return avg_loss
end

return Tester

