-- Michael B Chang

-- Third Party Imports
require 'torch'
require 'nn'
require 'optim'
require 'image'
require 'xlua'
require 'Base'
require 'sys'
require 'rmsprop'
require 'pl'

-- Local Imports
local model_utils = require 'model_utils'
local D = require 'DataLoader'
local M = require 'model_new'
require 'logging_utils'


-- TODO: Have a shell script for openmind and for pc

-- missing params: seed, experiment_string
-- also booleans are strings somehow
-- Note that the dataloaders will reset their current_batch after they've gone through all their batches.
--TODO what? optim.rmsprop is different from rmsprop?
------------------------------------- Init -------------------------------------

mp = lapp[[
   -d,--root          (default "logs")      	subdirectory to save logs
   -m,--model         (default "lstm")   		type of model tor train: lstm |
   -p,--plot                                	plot while training
   -c,--server		  (default "pc")			pc=personal | op = openmind
   -s,--shuffle  	  (default false)
   -r,--lr            (default 0.0005)      	learning rate
   -i,--max_epochs    (default 5)           	maximum nb of iterations per batch, for LBFGS
   --batch_size       (default 1)           	1 pc, 100 op
   --rnn_dim          (default 100)
   --seq_length       (default 10)				10 pc, 20 op
   --layers           (default 4)
   --rand_init_wts    (default false)
   --max_grad_norm    (default 10)
   --save_every		  (default 50)
   --print_every	  (default 10)
]]

if mp.server == 'pc' then
	mp.winsize = 10
	mp.dataset_folder = 'hey'
	mp.batch_size = 1
	mp.seq_length = 10
	mp.num_threads = 1
    mp.plot = false
	mp.cuda = false
	mp.cunn = false
else
	mp.winsize = 20
	mp.dataset_folder = '/om/user/mbchang/physics-data/dataset_files'
	mp.batch_size = 100
	mp.seq_length = 20
	mp.num_threads = 4
    mp.plot = false
	mp.cuda = true
	mp.cunn = true
end

mp.input_dim = 8.0*mp.winsize/2
mp.out_dim = 8.0*mp.winsize/2
mp.descrip = create_experiment_string({'batch_size', 'seq_length', 'layers', 'rnn_dim', 'max_epochs'}, mp) .. 'main_test'
mp.savedir = mp.root .. '/' .. mp.descrip

if mp.rand_init_wts then torch.manualSeed(123) end
if mp.shuffle == 'false' then mp.shuffle = false end
if mp.rand_init_wts == 'false' then mp.rand_init_wts = false end
if mp.plot == 'false' then mp.plot = false end
if mp.cuda then require 'cutorch' end
if mp.cunn then require 'cunn' end

local optim_state = {learningRate   = mp.lr,
                     momentumDecay  = 0.1,
                     updateDecay    = 0.01}

local model, train_loader, test_loader

trainLogger = optim.Logger(paths.concat(mp.savedir ..'/', 'train.log'))
experimentLogger = optim.Logger(paths.concat(mp.savedir ..'/', 'experiment.log'))
if mp.plot == false then
    trainLogger.showPlot = false
    experimentLogger.showPlot = false
end

------------------------------- Helper Functions -------------------------------

-- initialize
function init(preload, model_path)
    print("Network parameters:")
    print(mp)
    local data_loader_args = {mp.dataset_folder,
                              mp.batch_size,
                              mp.shuffle,
                              mp.cuda}
    train_loader = D.create('trainset', {}, unpack(data_loader_args))
    test_loader = D.create('testset', {}, unpack(data_loader_args))
    model = M.create(mp, preload, model_path)
    local epoch = 0  -- TODO Not sure if this is necessary
    local beginning_time = torch.tic() -- TODO not sure if this is necessary
    local start_time = torch.tic()  -- TODO not sure if this is necessary
    print("Initialized Network")
end

-- closure: returns loss, grad_params
function feval_train(params_)  -- params_ should be first argument
    local this, context, y, mask = unpack(train_loader:next_batch()) -- TODO this should take care of curriculum, how to deal with dataloader?
    local loss, state, predictions = model:fp(params_, {this=this,context=context}, y)
    local grad = model:bp({this=this,context=context}, y, mask, state)
    collectgarbage()
    return loss, grad -- f(x), df/dx
end

-- trains for one epoch
function train(epoch_num)
    local cntr = 0, new_params, train_loss
    for t = 1,train_loader.num_batches do
        xlua.progress(t, train_loader.num_batches)
        new_params, train_loss = rmsprop(feval_train, model.theta.params, optim_state)  -- next batch
        assert(new_params == model.theta.params)
        -- if t % mp.print_every == 0 then
        --     print(string.format("epoch %2d\titeration %2d\tloss = %6.8f\tgradnorm = %6.4e", epoch_num, t, train_loss[1], model.theta.grad_params:norm()))
        -- end

        trainLogger:add{['MSE loss (train set)'] =  train_loss[1]}
        trainLogger:style{['MSE loss (train set)'] = '-'}
        trainLogger:plot()

        cntr = cntr + 1
        if mp.cuda then cutorch.synchronize() end
        collectgarbage()
    end
    return train_loss[1]  -- because train_loss is returned as a table
end

-- test on dataset
function test(dataloader)
    local sum_loss = 0
    for i = 1,dataloader.num_batches do
        xlua.progress(i, dataloader.num_batches)
        local this, context, y, mask = unpack(dataloader:next_batch()) -- TODO this should take care of curriculum, how to deal with dataloader?
        local test_loss, state, predictions = model:fp(model.theta.params, {this=this,context=context}, y)
        sum_loss = sum_loss + test_loss

        -- here you have the option to save predictions into a file
        local prediction = predictions[torch.find(mask,1)[1]] -- (1, windowsize/2)

        -- reshape to -- (num_samples x windowsize/2 x 8)
        prediction = prediction:reshape(this:size(1),
                                        mp.winsize/2,
                                        dataloader.object_dim)

        -- if saveoutput then
        --     assert(torch.type(model)=='string')
        --     self:save_example_prediction({this, context, y, prediction, context_future},
        --                         {config, start, finish},
        --                         {'model_predictions', model})
        -- end
    end
    local avg_loss = sum_loss/dataloader.num_batches
    collectgarbage()
    return avg_loss
end

-- runs experiment
function experiment()
    torch.setnumthreads(mp.num_threads)
    print('<torch> set nb of threads to ' .. torch.getnumthreads())
    for i = 1, mp.max_epochs do
        if i == 10 then assert(false) end
        local train_loss
        train_loss = train(i)
        -- train_loss = test(train_test_loader)
        -- print('train loss\t', train_loss)

        local dev_loss = test(test_loader)
        -- print('avg dev loss\t', dev_loss)

        -- Save logs
        experimentLogger:add{['MSE loss (train set)'] =  train_loss,
                             ['MSE loss (test set)'] =  dev_loss}
        experimentLogger:style{['MSE loss (train set)'] = '-',
                               ['MSE loss (test set)'] = '-'}
        experimentLogger:plot()

        -- Save network
        torch.save(model.savedir .. '/network.t7', model.network)
        torch.save(model.savedir .. '/params.t7', model.theta.params)

        if mp.cuda then cutorch.synchronize() end
        collectgarbage()
    end
end

------------------------------------- Main -------------------------------------
init(false)
experiment()
