-- Train DDCIGN 

require 'metaparams'
require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'model'
require 'image'
require 'rmsprop'

if common_mp.cuda then
    require 'cutorch'
    require 'cunn'
end

if common_mp.cudnn then
    require 'cudnn'
end

local DataLoader = require 'DataLoader'
local model_utils = require 'model_utils'

local Trainer = {}
Trainer.__index = Trainer

function Trainer.create(dataset, mp)
    local self = {}
    setmetatable(self, Trainer)
    self.mp = mp
    self.dataset = dataset  -- string name of folder containing trainig examples
    self.train_loader = DataLoader.create(self.dataset, self.mp.dataset_folder, self.mp.shuffle)
    collectgarbage()
    return self
end

function Trainer:prepare_logs(learning_rate)
    self.mp.learning_rate = learning_rate
    self.logs = {}
    self.cmd = torch.CmdLine()
    self.logs.savefile = common_mp.results_folder .. '/saved_model,lr=' .. self.mp.learning_rate
    self.logs.savefile = self.logs.savefile .. '.t7'
    self.logs.lossesfile = common_mp.results_folder .. '/losses,lr=' .. self.mp.learning_rate .. '_results.t7'
    self.logs.train_losses = {losses={}, grad_norms={}}
    collectgarbage()
end

function Trainer:create_model()
    self.network = init_network(self.mp)
    if common_mp.cuda then self.network:cuda() end

    ------------------------------------- Parameters -------------------------------------
    self.theta = {}
    self.theta.params, self.theta.grad_params = self.network:getParameters()
    -- self.theta.params, self.theta.grad_params = model_utils.combine_all_parameters(self.protos.encoder, self.protos.lstm, self.protos.decoder) 
    print('self.theta.params', #self.theta.params)

    ------------------------------------ Clone Model -------------------------------------
    self.rnns = g_cloneManyTimes(self.network, self.mp.seq_length, not network.parameters)

    -------------------------------- Initialize LSTM State -------------------------------
    -- This will cache the values that s takes on in one forward pass
    self.s = {}
    for j = 0, self.mp.seq_length do
        self.s[j] = {}
        for d = 1, 2 * self.mp.layers do
            self.s[j][d] = model_utils.transfer_data(torch.zeros(torch.Tensor(self.mp.batch_size, self.mp.rnn_dim)), common_mp.cuda) 
        end
    end
    -- This will cache the values of the grad of the s 
    self.ds = {}
    for d = 1, 2 * params.layers do
        self.ds[d] = model_utils.transfer_data(torch.zeros(self.mp.batch_size, self.mp.rnn_dim))
    end

    -- This will cache the err
    self.err = model_utils.transfer_data(torch.zeros(self.mp.seq_length))

    collectgarbage()
end


function Trainer:reset_state()
    for j = 0, self.mp.seq_length do
        for d = 1, 2 * self.mp.layers do
            self.s[j][d]:zero()
        end
    end
end

function Trainer:reset_ds()
    for d = 1, #self.ds do
        self.ds[d]:zero()
    end
end

-- reset state before the forward pass
-- reset ds before the backward pass


function Trainer:forward_pass_train(params_, x, y)
    -- x is a table!
    if params_ ~= self.theta.params then self.theta.params:copy(params_) end
    self.theta.grad_params:zero()  -- reset gradient
    self:reset_state()  -- reset s

    -- unpack inputs
    local this_past     = x.this:clone()
    local context       = x.context:clone()
    local this_future   = y:clone()

    assert(this_past:size(1) == self.mp.batch_size and this_past:size(2) == self.mp.input_dim)
    assert(context:size(1) == self.mp.batch_size and context:size(2)==self.mp.seq_length
            and context:size(3) == self.mp.input_dim)
    assert(this_future:size(1) == self.mp.batch_size and this_future:size(2) == self.mp.input_dim)

    local accum_loss = 0
    local loss = {}
    local predictions = {}
    for i = 1, seq_length do
        local sim1 = self.s[i-1]  -- had been reset to 0 for initial pass
        loss[i], self.s[i], predictions[i] = unpack(self.rnns[i]:forward({this_past, context[{{},i}], sim1, this_future}))  -- problem! (feeding thisp_future every time; is that okay because I just update the gradient based on desired timesstep?)
        accum_loss = accum_loss + loss[i]
    end 

    collectgarbage()
    return accum_loss, self.s, predictions
end


function Trainer:backward_pass_train(x, y, mask, loss, state, predictions)
    -- assert that state equals self.s
    for j = 0, self.mp.seq_length do
        for d = 1, 2 * self.mp.layers do
            assert(torch.sum(state:eq(self.s)) == torch.numel(self.s))
        end
    end 

    self.theta.grad_params:zero()
    self.reset_ds()

    -- unpack inputs. All of these have been CUDAed already if need be
    local this_past     = x.this:clone()
    local context       = x.context:clone()
    local this_future   = y:clone()

    for i = self.mp.params.seq_length, 1, -1 do
        local sim1 = state[i - 1]
        local derr = mask:clone()[i]
        local dpred = transfer_data(torch.zeros(self.mp.batch_size,self.mp.out_dim))
        local dnewx = transfer_data(torch.zeros(params.bsize, 32, 32))
        local dpast, self.ds = unpack(self.rnns[i]:backward({this_past, context, sim1}, {derr, self.ds, dpred}))
        -- cutorch.synchronize()
    end
    self.theta.grad_params:clamp(-self.mp.max_grad_norm, self.mp.max_grad_norm)
    collectgarbage()
    return loss, self.theta.grad_params
end


-- function Trainer:backward_pass_train(x, y, loss, embeddings, lstm_c, lstm_h, predictions)
--     local dembeddings = {}
--     local dlstm_c = {[self.mp.seq_length]=self.lstm_init_state.dfinalstate_c}    -- internal cell states of LSTM
--     local dlstm_h = {}                                  -- output values of LSTM

--     for t = self.mp.seq_length, 1, -1 do
--         local doutput_t = self.clones.criterion[t]:backward(predictions[t], torch.squeeze(y[{{},{t}}]))

--         if t == self.mp.seq_length then
--             assert(dlstm_h[t] == nil)
--             dlstm_h[t] = self.clones.decoder[t]:backward(lstm_h[t], doutput_t)
--         else
--             dlstm_h[t]:add(self.clones.decoder[t]:backward(lstm_h[t], doutput_t))
--         end

--         dembeddings[t], dlstm_c[t-1], dlstm_h[t-1] = unpack(self.clones.lstm[t]:backward(
--             {embeddings[t], lstm_c[t-1], lstm_h[t-1]}, -- input to the lstm
--             {dlstm_c[t], dlstm_h[t]}  -- output of the lstm 
--         ))

--         self.clones.encoder[t]:backward(torch.squeeze(x[{{},{t}}]), dembeddings[t])
--         collectgarbage()
--     end

--     self.lstm_init_state.initstate_c:copy(lstm_c[#lstm_c])
--     self.lstm_init_state.initstate_h:copy(lstm_h[#lstm_h])

--     self.theta.grad_params:clamp(-self.mp.max_grad_norm, self.mp.max_grad_norm)
--     collectgarbage()
--     return loss, self.theta.grad_params
-- end




function Trainer:reset(learning_rate)
    self:prepare_logs(learning_rate)
    self:create_model()  -- maybe put this into constructor
end


function Trainer:train(num_iters, epoch_num)


    function feval_train(params_)
        -- feval MUST return loss, grad_loss in order to get fed into the optimizer!
        local this, context, y, mask = unpack(self.train_loader:next_batch())  -- the way it is defined in loader is to just keep cycling through the same dataset
        local train_loss, state, predictions = self:forward_pass_train(params_, {this,context}, y)
        local loss, grad_loss = self:backward_pass_train({this,context}, y, mask, train_loss, , state, predictions)
        assert(loss == train_loss)
        collectgarbage()
        return loss, grad_loss
    end

    -- function feval_train(params_)
    --     -- feval MUST return loss, grad_loss in order to get fed into the optimizer!
    --     local x, y = self.train_loader:next_batch(self.mp)  -- the way it is defined in loader is to just keep cycling through the same dataset
    --     local train_loss, embeddings, lstm_c, lstm_h, predictions = self:forward_pass_train(params_, x, y)
    --     local loss, grad_loss = self:backward_pass_train(x, y, train_loss, embeddings, lstm_c, lstm_h, predictions)
    --     assert(loss == train_loss)
    --     collectgarbage()
    --     return loss, grad_loss
    -- end

    -- here do epoch training
    local optim_state = {learningRate = self.mp.learning_rate}
    for i = 1,num_iters do 
        local _, loss = rmsprop(feval_train, self.theta.params, optim_state)  -- this is where the training actually happens
        self.logs.train_losses.losses[#self.logs.train_losses.losses+1] = loss[1]
        self.logs.train_losses.grad_norms[#self.logs.train_losses.grad_norms+1] = self.theta.grad_params:norm()

        if i % self.mp.print_every == 0 then
            print(string.format("epoch %2d\titeration %2d\tloss = %6.8f\tgradnorm = %6.4e", epoch_num, i, loss[1], self.theta.grad_params:norm()))
        end

        if i % self.mp.save_every == 0 then 
            torch.save(self.logs.savefile, self.protos)
            torch.save(self.logs.lossesfile, self.logs.train_losses)
            print('saved model')
        end
        collectgarbage()
    end
    torch.save(self.logs.savefile, self.protos)
    torch.save(self.logs.lossesfile, self.logs.train_losses)
    return self.logs.train_losses.losses[#self.logs.train_losses.losses], self.protos --self.logs.savefile
end    

-- train_mp.learning_rate = 5e-4
-- torch.manualSeed(123)
-- trainer = Trainer.create('train', train_mp, train_mp_ignore)
-- trainer:reset(5e-4)
-- final_loss = trainer:train(200, 0)
-- print(final_loss)

return Trainer



