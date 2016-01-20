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
   -d,--root          (default "logs2")      	subdirectory to save logs
   -m,--model         (default "lstm")   		type of model tor train: lstm |
   -n,--name          (default "lalala")
   -p,--plot          (default true)                    	plot while training
   -j,--traincfgs     (default "")
   -k,--testcfgs      (default "")
   -o,--opt           (default "rmsprop")       rmsprop | adam | optimrmsprop
   -c,--server		  (default "op")			pc=personal | op = openmind
   -s,--shuffle  	  (default false)
   -r,--lr            (default 0.0005)      	learning rate
   -i,--max_epochs    (default 50)           	maximum nb of iterations per batch, for LBFGS
   --rnn_dim          (default 100)
   --layers           (default 4)
   --seed             (default true)
   --max_grad_norm    (default 10)
   --save_output	  (default false)
   --print_every      (default 10)
]]

if mp.server == 'pc' then
    mp.root = 'logs'
	mp.winsize = 10  --10
	mp.dataset_folder = 'hey'
    mp.traincfgs = ''--[1-1-0]'
    mp.testcfgs = ''--[1-1-0]'
	mp.batch_size = 1
	mp.seq_length = 10
	mp.num_threads = 1
    mp.plot = true
	mp.cuda = false
	mp.cunn = false
else
	mp.winsize = 20
	mp.dataset_folder = '/om/data/public/mbchang/physics-data/dataset_files_subsampled'
	mp.batch_size = 50  -- this is decided by looking at the dataset
	mp.seq_length = 10
	mp.num_threads = 4
    mp.plot = false
	mp.cuda = true
	mp.cunn = true
end

mp.input_dim = 8.0*mp.winsize/2
mp.out_dim = 8.0*mp.winsize/2
-- mp.descrip = create_experiment_string({'batch_size', 'seq_length', 'layers', 'rnn_dim', 'max_epochs'}, mp)
mp.savedir = mp.root .. '/' .. mp.name

-- TODO: write a function to convert "false" to false
if mp.seed then torch.manualSeed(123) end
if mp.shuffle == 'false' then mp.shuffle = false end
if mp.rand_init_wts == 'false' then mp.rand_init_wts = false end
if mp.save_output == 'false' then mp.save_output = false end
if mp.plot == 'false' then mp.plot = false end
if mp.cuda then require 'cutorch' end
if mp.cunn then require 'cunn' end


local optimizer, optim_state
if mp.opt == 'rmsprop' then
    optimizer = rmsprop
    optim_state = {learningRate   = mp.lr,
                   momentumDecay  = 0.1,
                   updateDecay    = 0.01}
elseif mp.opt == 'optimrmsprop' then
    optimizer = optim.rmsprop
    optim_state = {learningRate   = mp.lr}
elseif mp.opt == 'adam' then
    optimizer = optim.adam
    optim_state = {learningRate   = mp.lr}
end


local model, train_loader, test_loader, modelfile

------------------------------- Helper Functions -------------------------------

-- initialize
function inittrain(preload, model_path)
    print("Network parameters:")
    print(mp)
    local data_loader_args = {mp.dataset_folder,
                              mp.batch_size,
                              mp.shuffle,
                              mp.cuda}
    train_loader = D.create('trainset', D.convert2allconfigs(mp.traincfgs), unpack(data_loader_args))
    val_loader =  D.create('valset', D.convert2allconfigs(mp.testcfgs), unpack(data_loader_args))  -- using testcfgs
    test_loader = D.create('testset', D.convert2allconfigs(mp.testcfgs), unpack(data_loader_args))
    model = M.create(mp, preload, model_path)

    trainLogger = optim.Logger(paths.concat(mp.savedir ..'/', 'train.log'))
    experimentLogger = optim.Logger(paths.concat(mp.savedir ..'/', 'experiment.log'))
    if mp.plot == false then
        trainLogger.showPlot = false
        experimentLogger.showPlot = false
    end
    print("Initialized Network")
end

function inittest(preload, model_path)
    print("Network parameters:")
    print(mp)
    local data_loader_args = {mp.dataset_folder,
                              mp.batch_size,
                              mp.shuffle,
                              mp.cuda}
    test_loader = D.create('testset', D.convert2allconfigs(mp.testcfgs), unpack(data_loader_args))
    model = M.create(mp, preload, model_path)
    modelfile = model_path
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
        -- xlua.progress(t, train_loader.num_batches)
        new_params, train_loss = optimizer(feval_train, model.theta.params, optim_state)  -- next batch
        assert(new_params == model.theta.params)
        if t % mp.print_every == 0 then
            print(string.format("epoch %2d\titeration %2d\tloss = %6.8f\tgradnorm = %6.4e",
                    epoch_num, t, train_loss[1], model.theta.grad_params:norm()))
        end

        trainLogger:add{['log MSE loss (train set)'] =  torch.log(train_loss[1])}
        trainLogger:style{['log MSE loss (train set)'] = '~'}
        if mp.plot then trainLogger:plot() end

        cntr = cntr + 1
        if mp.cuda then cutorch.synchronize() end
        collectgarbage()
    end
    return train_loss[1]  -- because train_loss is returned as a table
end

-- test on dataset
function test(dataloader, params_, saveoutput)
    local sum_loss = 0
    for i = 1,dataloader.num_batches do
        if mp.server == 'pc ' then xlua.progress(i, dataloader.num_batches) end
        local this, context, y, mask, config, start, finish, context_future = unpack(dataloader:next_batch()) -- TODO this should take care of curriculum, how to deal with dataloader?
        local test_loss, state, predictions = model:fp(params_, {this=this,context=context}, y)
        sum_loss = sum_loss + test_loss

        -- here you have the option to save predictions into a file
        local prediction = predictions[torch.find(mask,1)[1]] -- (1, windowsize/2)

        -- reshape to -- (num_samples x windowsize/2 x 8)
        prediction = prediction:reshape(this:size(1),
                                        mp.winsize/2,
                                        dataloader.object_dim)

        if saveoutput then
            save_example_prediction({this, context, y, prediction, context_future},
                                    {config, start, finish},
                                    modelfile,
                                    dataloader)
        end
    end
    local avg_loss = sum_loss/dataloader.num_batches
    collectgarbage()
    return avg_loss
end

function save_example_prediction(example, description, modelfile_, dataloader)
    --[[
        example: {this, context, y, prediction, context_future}
        description: {config, start, finish}
        modelfile_: like '/Users/MichaelChang/Documents/Researchlink/SuperUROP/Code/dynamics/logs/lalala/network.t7'

        will save to something like:
            logs/<experiment-name>/predictions/<config.h5>
    --]]

    --unpack
    local this, context, y, prediction, context_future = unpack(example)
    local config, start, finish = unpack(description)

    local subfolder = mp.savedir .. '/' .. 'predictions/'
    if not paths.dirp(subfolder) then paths.mkdir(subfolder) end
    local save_path = mp.savedir .. '/' .. 'predictions/' .. config..'_['..start..','..finish..'].h5'

    if mp.cuda then
        prediction = prediction:float()
        this = this:float()
        context = context:float()
        y = y:float()
        context_future = context_future:float()
    end

    local num_past = math.floor(mp.winsize/2)
    local num_future = mp.winsize-math.floor(mp.winsize/2)

    -- For now, just save it as hdf5. You can feed it back in later if you'd like
    save_to_hdf5(save_path,
        {pred=prediction,
        this=this:reshape(this:size(1),
                    num_past,
                    dataloader.object_dim),
        context=context:reshape(context:size(1),
                    context:size(2),
                    num_past,
                    dataloader.object_dim),
        y=y:reshape(y:size(1),
                    num_past,
                    dataloader.object_dim),
        context_future=context_future:reshape(context_future:size(1),
                    context_future:size(2),
                    num_future,
                    dataloader.object_dim)})
end

-- runs experiment
function experiment()
    torch.setnumthreads(mp.num_threads)
    print('<torch> set nb of threads to ' .. torch.getnumthreads())
    for i = 1, mp.max_epochs do
        checkpoint(mp.savedir .. '/network.t7', model.network, mp) -- model.rnns[1]?
        -- checkpoint(mp.savedir .. '/rnn1.t7', model.rnns[1], mp) -- model.rnns[1]?
        checkpoint(mp.savedir .. '/params.t7', model.theta.params, mp)
        print('Saved model')

        local train_loss
        train_loss = train(i)
        -- train_loss = test(train_test_loader)
        local val_loss = test(val_loader, model.theta.params, false)
        local test_loss = test(test_loader, model.theta.params, false)

        -- Save logs
        experimentLogger:add{['log MSE loss (train set)'] =  torch.log(train_loss),
                             ['log MSE loss (val set)'] =  torch.log(val_loss),
                             ['log MSE loss (test set)'] =  torch.log(test_loss)}
        experimentLogger:style{['log MSE loss (train set)'] = '~',
                               ['log MSE loss (val set)'] = '~',
                               ['log MSE loss (test set)'] = '~'}
        if mp.plot then experimentLogger:plot() end
        if mp.cuda then cutorch.synchronize() end
        collectgarbage()
    end
end

function checkpoint(savefile, data, mp_)
    if mp_.cuda then
        print('converting to float')
        data = data:float()
        torch.save(savefile, data)
        data = data:cuda()
    else
        torch.save(savefile, data)
    end
    collectgarbage()
end

------------------------------------- Main -------------------------------------
-- inittrain(false)
-- experiment()

local exp_folder = '/home/mbchang/code/physics_engine/logs2/baselinesubsampled_opt_adam_lr_0.001'
-- local exp_folder = ''/Users/MichaelChang/Documents/Researchlink/SuperUROP/Code/dynamics/logs/lalala'
inittest(true, exp_folder ..'/'..'network.t7')
print(test(test_loader, torch.load(exp_folder..'/'..'params.t7'), false))
