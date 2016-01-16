require 'torch'
local model_utils = require 'model_utils'

require 'nn'
require 'optim'
require 'image'
require 'xlua'
require 'Base'
require 'sys'
require 'rmsprop'
require 'pl'


-- TODO: Have a shell script for openmind and for pc

-- missing params: seed, experiment_string
-- also booleans are strings somehow
-- you can
--------------------------- Init ------------------------------
require 'logging_utils'


mp = lapp[[
   -d,--savedir       (default "logs")      	subdirectory to save logs
   -m,--model         (default "lstm")   		type of model tor train: lstm |
   -p,--plot                                	plot while training
   -c,--server		  (default "pc")			pc=personal | op = openmind
   -s,--shuffle  	  (default false)
   -r,--lr            (default 0.0005)      	learning rate
   -i,--max_epochs    (default 50)           	maximum nb of iterations per batch, for LBFGS
   --batch_size       (default 1)           	1 pc, 100 op
   --rnn_dim          (default 100)
   --seq_length       (default 10)				10 pc, 20 op
   --layers           (default 4)
   --rand_init_wts    (default false)
   --max_grad_norm    (default 10)
   --save_every		  (default 50)
   --print_every	  (default 10)
]]


-- torch.manualSeed(123)
-- if params.rand_init_wts then torch.manualSeed(123) end
if mp.shuffle == 'false' then mp.shuffle = false end
if mp.rand_init_wts == 'false' then mp.rand_init_wts = false end

if mp.server == 'pc' then
	mp.winsize = 10
	mp.dataset_folder = 'hey'
	mp.batch_size = 1
	mp.seq_length = 10
	mp.num_threads = 1
	mp.cuda = false
	mp.cunn = false
else
	mp.winsize = 20
	mp.dataset_folder = '/om/user/mbchang/physics-data/dataset_files'
	mp.batch_size = 100
	mp.seq_length = 20
	mp.num_threads = 4
	mp.cuda = true
	mp.cunn = true
end

mp.input_dim = 8.0*mp.winsize/2
mp.out_dim = 8.0*mp.winsize/2
mp.results_folder = create_experiment_string({'batch_size', 'seq_length', 'layers', 'rnn_dim', 'max_epochs'}, mp) .. 'new_test'

local optim_state = {learningRate   = mp.lr,
                     momentumDecay  = 0.1,
                     updateDecay    = 0.01}

local D = require 'DataLoader'
local M = require 'model_new'
local model, train_loader, test_loader

-- threads
-- torch.setnumthreads(common_mp.num_threads)
-- print('<torch> set nb of threads to ' .. torch.getnumthreads())

-- local Tester = require 'test'
-- local Trainer = require 'train'


common_mp = tablex.deepcopy(mp)
train_mp = tablex.deepcopy(mp)  -- problem: something in train_mp got mutated in Trainer
test_mp = tablex.deepcopy(mp)

if common_mp.cuda then require 'cutorch' end
if common_mp.cunn then require 'cunn' end


-- local trainer = Trainer.create('trainset', train_mp)  -- need to specify learning rate here
-- local trainer_tester = Tester.create('trainset', test_mp)
-- local tester = Tester.create('testset', test_mp)

-- local learning_rates = {5e-4, 5e-5, 5e-6}

local experiment_results = common_mp.results_folder .. '/experiment_results.t7'

-- if not common_mp.rand_init_wts then torch.manualSeed(123) end
torch.manualSeed(123)

-- initialize
function init(preload, model_path)
  print("Network parameters:")
  print(mp)
  train_loader = D.create('trainset', mp.dataset_folder, {}, mp.batch_size, mp.curriculum, mp.shuffle)  -- TODO: take care of all curriculum here
  test_loader = D.create('testset', mp.dataset_folder, {}, mp.batch_size, mp.curriculum, mp.shuffle)  -- TODO: take care of all curriculum here
  model = M.create(mp, preload, model_path)
  local epoch = 0  -- TODO Not sure if this is necessary
  local beginning_time = torch.tic() -- TODO not sure if this is necessary
  local start_time = torch.tic()  -- TODO not sure if this is necessary
  print("Initialized Network")
end

-- closure
-- returns loss, grad_params
function feval_train(params_)  -- params_ should be first argument
    local this, context, y, mask = unpack(train_loader:next_batch()) -- TODO this should take care of curriculum, how to deal with dataloader?
    local loss, state, predictions = model:fp(params_, {this=this,context=context}, y)
    local grad = model:bp({this=this,context=context}, y, mask, state)
    collectgarbage()
    return loss, grad -- f(x), df/dx
end

-- closure
-- returns loss, grad_params
function feval_test(params_)  -- params_ should be first argument
    local this, context, y, mask = unpack(test_loader:next_batch()) -- TODO this should take care of curriculum, how to deal with dataloader?
    local loss, state, predictions = model:fp(params_, {this=this,context=context}, y)
    local grad = model:bp({this=this,context=context}, y, mask, state)
    collectgarbage()
    return loss, grad -- f(x), df/dx
end

--TODO what? optim.rmsprop is different from rmsprop?

function train(epoch_num)
    local cntr = 0, new_params, train_loss
    for t = 1,train_loader.num_batches do  -- TODO
        -- if t == 2 then
        --     print('params', model.theta.params)
        --     assert(false)
        -- end
        -- local optim_state = {learningRate   = mp.lr,
        --                      momentumDecay  = 0.1,
        --                      updateDecay    = 0.01}
        -- print('optim_state before', optim_state)
        -- assert(false)

        --   xlua.progress(t, dataloader.num_batches)
        new_params, train_loss = rmsprop(feval_train, model.theta.params, optim_state)  -- next batch
        assert(new_params == model.theta.params)  -- TODO remove later
        --   if params.plot and math.fmod(cntr, 20) == 0 then test() end

        -- print('params', model.theta.params)
        -- assert(false)

        if t % mp.print_every == 0 then
            print(string.format("epoch %2d\titeration %2d\tloss = %6.8f\tgradnorm = %6.4e", epoch_num, t, train_loss[1], model.theta.grad_params:norm()))
        end


        if t % mp.save_every == 0 then
            -- convert from cuda to float before saving
            -- print('common_mp.cuda', common_mp.cuda)
            -- if mp.cuda then
            --     -- print('Converting to float before saving')
            --     model.network:float()
            --     torch.save(self.logs.savefile, self.network)
            --     self.network:cuda()
            -- else
            --     torch.save(self.logs.savefile, self.network)
            -- end
            print('saved model')
            -- torch.save(self.logs.lossesfile, self.logs.train_losses)
        end


        cntr = cntr + 1
        -- trainLogger:plot()
        if common_mp.cuda then cutorch.synchronize() end
        collectgarbage()
    end
    return train_loss
end


function test()
    local sum_loss = 0
    for i = 1,test_loader.num_batches do
        -- xlua.progress(t, test_loader.num_batches)
        local this, context, y, mask = unpack(test_loader:next_batch()) -- TODO this should take care of curriculum, how to deal with dataloader?
        local loss, state, predictions = model:fp(params_, {this=this,context=context}, y)
        sum_loss = sum_loss + test_loss

        -- here you have the option to save predictions into a file
        local prediction = all_preds[torch.find(mask,1)[1]] -- (1, windowsize/2)
        -- this would be your 'this', or you could shift over, or do other interesting things

        -- reshape to -- (num_samples x windowsize/2 x 8)
        prediction = prediction:reshape(this:size(1), self.mp.winsize/2, self.test_loader.object_dim)

        -- if saveoutput then
        --     assert(torch.type(model)=='string')
        --     self:save_example_prediction({this, context, y, prediction, context_future},
        --                         {config, start, finish},
        --                         {'model_predictions', model})
        -- end
        local avg_loss = sum_loss/num_iters
        collectgarbage()
        return avg_loss
    end
end

init(false)



local all_results = {}

-- for index, learning_rate in pairs(learning_rates) do
local learning_rate = mp.lr
print('Learning rate:', learning_rate)
local train_losses = {}
local dev_losses = {}

local oldp, oldgp

for i = 1, mp.max_epochs do
    if i == 3 then assert(false) end

    -- Train
    -- this train_loss is the final loss after one epoch. We expect to see this go down as epochs increase
    local model
    local train_loss
    train_loss = train(i)  -- trainer.train_loader.num_batches
    -- Get the training loss
    -- local train_loss = trainer_tester:test(model, p, trainer_tester.test_loader.num_batches)  -- tester.test_loader.nbatches  -- creating new copy of model when I load into Tester!
    -- local train_loss = trainer_tester:test(model, trainer_tester.test_loader.num_batches)  -- tester.test_loader.nbatches  -- creating new copy of model when I load into Tester!
    print('train loss\t', train_loss)

    -- Test
    -- this train_loss is the final loss after one epoch. We expect to see this go in a parabola as epochs increase
    -- local dev_loss = tester:test(model, p, tester.test_loader.num_batches)  -- tester.test_loader.nbatches  -- creating new copy of model when I load into Tester!
    -- local dev_loss = tester:test(model, tester.test_loader.num_batches)  -- tester.test_loader.nbatches  -- creating new copy of model when I load into Tester!
    local dev_loss = test()

    -- Record loss
    train_losses[#train_losses+1] = train_loss
    dev_losses[#dev_losses+1] = dev_loss
    print('avg dev loss\t', dev_loss)
    -- When we save results, for this learning rate, this is the curve of train_loss and dev_loss as we go through epochs

    print('train_losses:', train_losses)
    print('dev_losses:', dev_losses)

    -- Can save here actually, so that you constantly save stuff. Basically you rewrite all_results[learning_rate] each time
    all_results[learning_rate] = {results_train_losses  = torch.Tensor(train_losses),
                                  results_dev_losses    = torch.Tensor(dev_losses)}
    print('all_results', all_results)
    torch.save(experiment_results, all_results)

    if common_mp.cuda then cutorch.synchronize() end
    collectgarbage()
end
-- end
