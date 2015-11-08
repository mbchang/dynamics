-- Train DDCIGN 

require 'metaparams'
require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'model'
require 'image'
require 'rmsprop'
require 'SteppableLSTM'

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

function Trainer.create(dataset, mp, mp_ignore)
    local self = {}
    setmetatable(self, Trainer)
    self.mp = mp
    self.mp_ignore = mp_ignore
    self.dataset = dataset  -- string name of folder containing trainig examples
    self.train_loader = DataLoader.create(self.dataset, self.mp.seq_length, self.mp.batch_size, self.mp.shuffle)
    collectgarbage()
    return self
end

function Trainer:prepare_logs(learning_rate)
    self.mp.learning_rate = learning_rate
    self.logs = {}
    self.cmd = torch.CmdLine()
    self.logs.savefile = common_mp.results_folder .. '/saved_model,lr=' .. self.mp.learning_rate
    -- self.logs.savefile = self.cmd:string(self.logs.savefile, self.mp, self.mp_ignore) .. '.t7'
    self.logs.savefile = self.logs.savefile .. '.t7'
    self.logs.lossesfile = common_mp.results_folder .. '/losses,lr=' .. self.mp.learning_rate .. '_results.t7'
    self.logs.train_losses = {losses={}, grad_norms={}}
    collectgarbage()
end

function Trainer:create_model()
    ------------------------------------ Create Model ------------------------------------
    self.protos        = {}
    self.protos.encoder      = init_baseline_encoder(self.mp, self.mp.LSTM_input_dim)
    self.protos.lstm         = init_baseline_lstm(self.mp, self.mp.LSTM_input_dim, self.mp.LSTM_hidden_dim)
    -- controller = nn.SteppableLSTM(vocab_size, vocab_size, opt.rnn_size, opt.num_layers, opt.dropout)
    self.protos.decoder      = init_baseline_decoder(self.mp, self.mp.LSTM_hidden_dim)
    self.protos.criterion    = nn.BCECriterion()

    if common_mp.cuda then 
        self.protos.encoder:cuda()
        self.protos.lstm:cuda()
        self.protos.decoder:cuda()
        self.protos.criterion:cuda()
    end

    ------------------------------------- Parameters -------------------------------------
    self.theta = {}
    self.theta.params, self.theta.grad_params = model_utils.combine_all_parameters(self.protos.encoder, self.protos.lstm, self.protos.decoder) 
    print('self.theta.params', #self.theta.params)
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

    -- LSTM final state's backward message (dloss/dfinalstate) is 0, since it doesn't influence predictions
    self.lstm_init_state.dfinalstate_c = self.lstm_init_state.initstate_c:clone()
    self.lstm_init_state.dfinalstate_h = self.lstm_init_state.initstate_c:clone()  -- actually not used

    if common_mp.cuda then
        self.lstm_init_state.initstate_c = self.lstm_init_state.initstate_c:cuda()
        self.lstm_init_state.initstate_h = self.lstm_init_state.initstate_h:cuda()
        self.lstm_init_state.dfinalstate_c = self.lstm_init_state.dfinalstate_c:cuda()
        self.lstm_init_state.dfinalstate_h = self.lstm_init_state.dfinalstate_h:cuda()
    end
    collectgarbage()
end

function Trainer:forward_pass_train(params_, x, y)
    if params_ ~= self.theta.params then
        self.theta.params:copy(params_)
    end
    self.theta.grad_params:zero()  -- reset gradient

    ------------------- forward pass -------------------
    local embeddings = {}
    local lstm_c = {[0]=self.lstm_init_state.initstate_c} -- internal cell states of LSTM
    local lstm_h = {[0]=self.lstm_init_state.initstate_h} -- output values of LSTM
    local predictions = {}

    local loss = 0
    for t = 1,self.mp.seq_length do
        embeddings[t] = self.clones.encoder[t]:forward(torch.squeeze(x[{{},{t}}]))
        lstm_c[t], lstm_h[t] = unpack(self.clones.lstm[t]:forward{embeddings[t], lstm_c[t-1], lstm_h[t-1]})
        predictions[t] = self.clones.decoder[t]:forward(lstm_h[t])
        loss = loss + self.clones.criterion[t]:forward(predictions[t], torch.squeeze(y[{{},{t}}]))

        -- DEBUGGING
        --image.display(predictions[t])  -- here you can also save the image
    end
    collectgarbage()
    return loss, embeddings, lstm_c, lstm_h, predictions
end

function Trainer:backward_pass_train(x, y, loss, embeddings, lstm_c, lstm_h, predictions)
    local dembeddings = {}
    local dlstm_c = {[self.mp.seq_length]=self.lstm_init_state.dfinalstate_c}    -- internal cell states of LSTM
    local dlstm_h = {}                                  -- output values of LSTM

    for t = self.mp.seq_length, 1, -1 do
        local doutput_t = self.clones.criterion[t]:backward(predictions[t], torch.squeeze(y[{{},{t}}]))

        if t == self.mp.seq_length then
            assert(dlstm_h[t] == nil)
            dlstm_h[t] = self.clones.decoder[t]:backward(lstm_h[t], doutput_t)
        else
            dlstm_h[t]:add(self.clones.decoder[t]:backward(lstm_h[t], doutput_t))
        end

        dembeddings[t], dlstm_c[t-1], dlstm_h[t-1] = unpack(self.clones.lstm[t]:backward(
            {embeddings[t], lstm_c[t-1], lstm_h[t-1]}, -- input to the lstm
            {dlstm_c[t], dlstm_h[t]}  -- output of the lstm 
        ))

        self.clones.encoder[t]:backward(torch.squeeze(x[{{},{t}}]), dembeddings[t])
        collectgarbage()
    end

    self.lstm_init_state.initstate_c:copy(lstm_c[#lstm_c])
    self.lstm_init_state.initstate_h:copy(lstm_h[#lstm_h])

    self.theta.grad_params:clamp(-self.mp.max_grad_norm, self.mp.max_grad_norm)
    collectgarbage()
    return loss, self.theta.grad_params
end

function Trainer:reset(learning_rate)
    self:prepare_logs(learning_rate)
    self:create_model()  -- maybe put this into constructor
end

function Trainer:train(num_iters, epoch_num)

    function feval_train(params_)
        -- feval MUST return loss, grad_loss in order to get fed into the optimizer!
        local x, y = self.train_loader:next_batch(self.mp)  -- the way it is defined in loader is to just keep cycling through the same dataset
        local train_loss, embeddings, lstm_c, lstm_h, predictions = self:forward_pass_train(params_, x, y)
        local loss, grad_loss = self:backward_pass_train(x, y, train_loss, embeddings, lstm_c, lstm_h, predictions)
        assert(loss == train_loss)
        collectgarbage()
        return loss, grad_loss
    end

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
    return self.logs.train_losses.losses[#self.logs.train_losses.losses], self.logs.savefile
end    

-- train_mp.learning_rate = 5e-4
-- torch.manualSeed(123)
-- trainer = Trainer.create('train', train_mp, train_mp_ignore)
-- trainer:reset(5e-4)
-- final_loss = trainer:train(200, 0)
-- print(final_loss)

return Trainer



