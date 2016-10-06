-- Michael B Chang

-- Third Party Imports
require 'torch'
require 'nn'
require 'optim'
require 'image'
require 'xlua'
require 'Base'
require 'sys'
require 'pl'
torch.setdefaulttensortype('torch.FloatTensor')
require 'data_utils'
local tablex = require 'pl.tablex'

-- Local Imports
local model_utils = require 'model_utils'
local D = require 'general_data_sampler'
local D2 = require 'datasaver'
require 'logging_utils'

config_args = require 'config'
local data_process = require 'data_process'

------------------------------------- Init -------------------------------------
local cmd = torch.CmdLine()
cmd:option('-mode', "exp", 'exp | pred | simulate | save')
cmd:option('-server', "op", 'pc = personal | op = openmind')
cmd:option('logs_root', 'logs', 'subdirectory to save logs and checkpoints')
cmd:option('data_root', '../data', 'subdirectory to save data')
cmd:option('-model', "ffobj", 'ff | ffobj | lstmobj | gruobj | cat | ind')
cmd:option('-name', "mj", 'experiment name')
cmd:option('-seed', 0, 'manual seed')

-- dataset
cmd:option('-dataset_folders', '', 'dataset folder')
cmd:option('-test_dataset_folders', '', 'dataset folder')

-- model params
cmd:option('-rnn_dim', 50, 'hidden dimension')
cmd:option('-nbrhd', false, 'restrict attention to neighborhood')
cmd:option('-nbrhdsize', 3.5, 'number of radii out to look. nbhrdsize of 2 is when they exactly touching')
cmd:option('-layers', 3, 'layers in network')
cmd:option('-relative', true, 'relative state vs absolute state')
cmd:option('-diff', false, 'use relative context position and velocity state')
cmd:option('-batch_norm', false, 'batch norm')
cmd:option('-num_past', 2, 'number of past timesteps')
cmd:option('-nlan', false, 'no look ahead for neighbors')

-- training options
cmd:option('-opt', "rmsprop", 'rmsprop | adam')
cmd:option('-batch_size', 50, 'batch size')
cmd:option('-shuffle', true, 'shuffle batches')
cmd:option('-max_iter', 1200000, 'max number of iterations (some huge number)')
cmd:option('-L2', 0, 'L2 regularization')  -- 0.001
cmd:option('-lr', 0.0003, 'learning rate')
cmd:option('-lrdecay', 0.99, 'learning rate annealing')
cmd:option('-val_window', 10, 'for testing convergence')
cmd:option('-val_eps', 1e-6, 'for testing convergence')  -- 1e-5
cmd:option('-im', false, 'infer mass')
cmd:option('-cf', false, 'collision filter')  -- should be on if -im is on
cmd:option('-vlambda', 1, 'velocity penalization')
cmd:option('-lambda', 1, 'angle penalization')

-- priority sampling
cmd:option('-ps', false, 'turn on priority sampling')
cmd:option('-rs', false, 'turn on random sampling')
cmd:option('-sharpen', 1, 'sharpen exponent')
cmd:option('-dropout', 0.0, 'dropout for lstm')

-- experiment options
cmd:option('-plot', false, 'turn on/off plot')

-- every options
cmd:option('-print_every', 500, 'print every number of batches')
cmd:option('-save_every', 100000, 'save every number of batches')  -- this should be every 100000
cmd:option('-val_every', 100000,'val every number of batches') -- this should be every 100000 
cmd:option('-lrdecay_every',2500,'decay lr every number of batches')
cmd:option('-lrdecayafter', 50000, 'number of epochs before turning down lr')
cmd:option('-cuda', false, 'gpu')
cmd:option('-fast', false, 'fast mode')

cmd:text()

-- parse input params
mp = cmd:parse(arg)

if mp.server == 'pc' then
    mp.data_root = 'mj_data'
    mp.logs_root = 'logs'
    mp.winsize = 3 -- total number of frames
    mp.num_past = 2 --10
    mp.num_future = 1 --10
	mp.batch_size = 5 --1
    mp.max_iter = 60 
    mp.nbrhd = true
    mp.lr = 3e-3
    mp.lrdecay = 0.5
    mp.lrdecayafter = 20
    mp.lrdecay_every = 20
    mp.layers = 5
    mp.rnn_dim = 24
    mp.model = 'bffobj'
    mp.im = false
    mp.cf = false
    mp.val_window = 5
    mp.val_eps = 2e-5
	-- mp.seq_length = 8 -- for the concatenate model
	mp.num_threads = 1
    mp.shuffle = false
    mp.batch_norm = false
    mp.print_every = 1
    mp.save_every = 20
    mp.val_every = 20
    mp.plot = true--true
	mp.cuda = false
    mp.rs = false
    mp.nlan = true
    mp.fast = true
else
	-- mp.winsize = 3  -- total number of frames
    -- mp.num_past = 2 -- total number of past frames
    mp.num_future = 1
	-- mp.seq_length = 8   -- for the concatenate model
	mp.num_threads = 4
end

local M

if mp.model == 'lstmobj' or mp.model == 'ffobj' or mp.model == 'gruobj' then
    M = require 'variable_obj_model'
elseif mp.model == 'bffobj' then 
    M = require 'branched_variable_obj_model'
elseif mp.model == 'cat' then 
    M = require 'concatenate'
elseif mp.model == 'ind' then 
    M = require 'independent'
elseif mp.model == 'np' then 
    M = require 'nop'
elseif mp.model == 'crnn' then 
    M = require 'clique_rnn'
elseif mp.model == 'lstmcat' then
    M = require 'lstm_model'
elseif mp.model == 'bl' then
    M = require 'blstm'
elseif mp.model == 'lstm' then
    M = require 'lstm'
elseif mp.model == 'ff' then
    M = require 'feed_forward_model'
else
    error('Unrecognized model')
end

mp.winsize = mp.num_past + mp.num_future
mp.object_dim = config_args.si.p[2]
mp.input_dim = mp.object_dim*mp.num_past
mp.out_dim = mp.object_dim*mp.num_future
if mp.model == 'crnn' then 
    mp.input_dim = mp.object_dim 
    mp.out_dim = mp.object_dim
end
mp.name = string.gsub(string.gsub(string.gsub(mp.name,'{',''),'}',''),"'",'')
mp.savedir = mp.logs_root .. '/' .. mp.name
print(mp.savedir)

torch.manualSeed(mp.seed)
if mp.cuda then
    require 'cutorch'
    require 'cunn'
    cutorch.manualSeed(mp.seed)
end

local optimizer, optim_state
if mp.opt == 'rmsprop' then
    optimizer = optim.rmsprop
    optim_state = {learningRate   = mp.lr}
elseif mp.opt == 'adam' then
    optimizer = optim.adam
    optim_state = {learningRate   = mp.lr}
else
    error('unknown optimizer')
end

mp.dataset_folders = assert(loadstring("return "..string.gsub(mp.dataset_folders,'\"',''))())
mp.test_dataset_folders = assert(loadstring("return "..string.gsub(mp.test_dataset_folders,'\"',''))())

local model, train_loader, test_loader, modelfile
local train_losses, val_losses, test_losses = {},{},{}

------------------------------- Helper Functions -------------------------------

-- initialize
function inittrain(preload, model_path, iters)
    print("Network parameters:")
    print(mp)
    if mp.cuda then
        require 'cutorch'
        require 'cunn'
    end
    local data_loader_args = {data_root=mp.data_root..'/',
                              dataset_folders=mp.dataset_folders,
                              maxwinsize=config_args.maxwinsize,
                              winsize=mp.winsize, -- not sure if this should be in mp
                              num_past=mp.num_past,
                              num_future=mp.num_future,
                              relative=mp.relative,
                              sim=false,
                              subdivide=config_args.subdivide,
                              shuffle=config_args.shuffle,
                              cuda=mp.cuda
                            }


    -- test_args is the same but with a different dataset_folder
    local test_args = tablex.deepcopy(data_loader_args)
    test_args.dataset_folders = mp.test_dataset_folders

    train_loader = D.create('trainset', tablex.deepcopy(data_loader_args))
    val_loader =  D.create('valset', tablex.deepcopy(data_loader_args))  -- using testcfgs
    test_loader = D.create('testset', tablex.deepcopy(test_args))
    train_test_loader = D.create('trainset', tablex.deepcopy(data_loader_args))
    model = M.create(mp, preload, model_path)

    local train_log_file
    if iters then
        train_log_file = 'train_'..iters..'.log'
    else
        train_log_file = 'train.log'
    end

    trainLogger = optim.Logger(paths.concat(mp.savedir ..'/', train_log_file))
    experimentLogger = optim.Logger(paths.concat(mp.savedir ..'/', 'experiment.log'))
    if mp.im then
        inferenceLogger = optim.Logger(paths.concat(mp.savedir ..'/', 'infer.log'))
    end
    if mp.plot == false then
        trainLogger.showPlot = false
        experimentLogger.showPlot = false
        if mp.im then
            inferenceLogger.showPlot = false
        end
    end

    -- save args
    local args_file
    if iters then
        args_file = mp.savedir..'/args'..iters..'.t7'
    else 
        args_file = mp.savedir..'/args.t7'
    end
    torch.save(args_file, {mp=mp,config_args=config_args})
    print("Initialized Network")
end

function initsavebatches()
    local wascudabefore = mp.cuda
    mp.cuda = false
    config_args.batch_size = mp.batch_size

    -- save training set
    for _, dataset_folder in pairs(mp.dataset_folders) do
        local data_folder = mp.data_root..'/'..dataset_folder..'/batches'
        if not paths.dirp(data_folder) then
            local jsonfolder = mp.data_root..'/'..dataset_folder..'/jsons'
            print('Saving batches of size '..mp.batch_size..' from '..jsonfolder..'into '..data_folder)
            local dp = data_process.create(jsonfolder, data_folder, config_args)
            dp:create_datasets_batches()
        else
            print('Batches for '..dataset_folder..' already made')
        end
    end

    -- save testing set
    for _, dataset_folder in pairs(mp.test_dataset_folders) do
        local data_folder = mp.data_root..'/'..dataset_folder..'/batches'
        if not paths.dirp(data_folder) then
            local jsonfolder = mp.data_root..'/'..dataset_folder..'/jsons'
            print('Saving batches of size '..mp.batch_size..' from '..jsonfolder..'into '..data_folder)
            local dp = data_process.create(jsonfolder, data_folder, config_args)
            dp:create_datasets_batches()
        else
            print('Batches for '..dataset_folder..' already made')
        end
    end

    if wascudabefore then mp.cuda = true end
end

-- closure: returns loss, grad_params
function feval_train(params_)
    local batch
    if mp.rs then
        batch = train_loader:sample_random_batch(mp.sharpen)
    else
        batch = train_loader:sample_priority_batch(mp.sharpen)
    end

    local loss, prediction = model:fp(params_, batch)
    local grad = model:bp(batch,prediction)

    if mp.L2 > 0 then
        -- Loss:
        loss = loss + mp.L2 * model.theta.params:norm(2)^2/2 
        -- Gradients:
        model.theta.grad_params:add(model.theta.params:clone():mul(mp.L2) )
    end

    train_loader:update_batch_weight(loss)
    if mp.cuda then cutorch.synchronize() end
    collectgarbage()
    return loss, grad -- f(x), df/dx
end

function train(start_iter, epoch_num)
    local epoch_num = epoch_num or 1
    local start_iter = start_iter or 1
    print('Start iter:', start_iter)
    print('Start epoch num:', epoch_num)

    -- Get the loss before training
    if start_iter == 1 then
        v_train_loss, v_val_loss, v_test_loss = validate()
        train_losses[#train_losses+1] = v_train_loss
        val_losses[#val_losses+1] = v_val_loss
        test_losses[#test_losses+1] = v_test_loss
    end

    for t = start_iter,mp.max_iter do

        local new_params, train_loss = optimizer(feval_train,
                                model.theta.params, optim_state)  -- next batch

        assert(new_params == model.theta.params)

        trainLogger:add{['log MSE loss (train set)'] = torch.log(train_loss[1])}
        trainLogger:style{['log MSE loss (train set)'] = '~'}

        if (t-start_iter+1) % mp.print_every == 0 then
            print(string.format("epoch %2d  iteration %2d  loss = %6.8f"..
                            "  gradnorm = %6.4e  batch = %d-%d    "..
                            "hardest batch: %d-%d    with loss %6.8f lr = %6.4e",
                    epoch_num, t, train_loss[1],
                    model.theta.grad_params:norm(),
                    train_loader.current_dataset,
                    train_loader.current_sampled_id,
                    train_loader:get_hardest_batch()[3],
                    train_loader:get_hardest_batch()[2],
                    train_loader:get_hardest_batch()[1],
                    optim_state.learningRate))
        end

        -- validate
        if (t-start_iter+1) % mp.val_every == 0 then
            v_train_loss, v_val_loss, v_test_loss = validate()
            train_losses[#train_losses+1] = v_train_loss
            val_losses[#val_losses+1] = v_val_loss
            test_losses[#test_losses+1] = v_test_loss
            assert(mp.save_every % mp.val_every == 0 or
                    mp.val_every % mp.save_every == 0)

            -- save
            if (t-start_iter+1) % mp.save_every == 0 then
                local model_file = string.format('%s/epoch%d_step%d_%.7f.t7',
                                            mp.savedir, epoch_num, t, v_val_loss)
                print('saving checkpoint to ' .. model_file)

                local checkpoint = {}
                checkpoint.model = model  -- TODO_lowpriority: should I save the model.theta?
                checkpoint.mp = mp
                checkpoint.train_losses = train_losses
                checkpoint.val_losses = val_losses
                checkpoint.test_losses = test_losses
                checkpoint.iters = t
                torch.save(model_file, checkpoint)
                print('Saved model')
            end

            -- here test for val_loss convergence
            if #val_losses >= mp.val_window then
                local val_loss_window = torch.Tensor(val_losses)[{{-mp.val_window,-1}}]
                -- these are torch Tensors
                local max_val_loss, max_val_loss_idx = torch.max(val_loss_window,1)
                local min_val_loss, min_val_loss_idx = torch.min(val_loss_window,1)

                local val_avg_delta = (val_loss_window[{{2,-1}}] - val_loss_window[{{1,-2}}]):mean()
                print('Average change in val loss over '..mp.val_window..
                        ' validations: '..val_avg_delta)
                -- test if the loss is going down. the average pairwise delta should be negative, and the last should be less than the first
                if val_avg_delta < 0 and torch.lt(max_val_loss_idx,min_val_loss_idx) then
                    print('Loss is decreasing')
                    -- if not we can lower the learning rate
                else
                    print('Loss is increasing')
                end

                print('Val loss difference in a window of '..
                        mp.val_window..': '..(max_val_loss-min_val_loss)[1])
                -- test if the max and min differ by less than epsilon
                print((max_val_loss-min_val_loss)[1])
                if (max_val_loss-min_val_loss)[1] < mp.val_eps then
                    print('That is less than '..mp.val_eps..'. Converged.')
                    break
                end
            end
        end

        -- lr decay
        -- here you can adjust the learning rate based on val loss
        if t >= mp.lrdecayafter and (t-start_iter+1) % mp.lrdecay_every == 0 then
            mp.lr = mp.lr*mp.lrdecay  -- I should mutate this because it should keep track of the current lr anyway
            optim_state.learningRate = mp.lr  
            print('Learning rate is now '..optim_state.learningRate)
        end

        if t % train_loader.num_batches == 0 then
            epoch_num = t / train_loader.num_batches + 1
        end

        if mp.plot then trainLogger:plot() end
        if mp.cuda then cutorch.synchronize() end
        collectgarbage()
    end
end

function test(dataloader, params_, saveoutput, num_batches)
    local sum_loss = 0
    local num_batches = num_batches or dataloader.num_batches

    if mp.fast then num_batches = math.min(5000, num_batches) end
    print('Testing '..num_batches..' batches')
    for i = 1,num_batches do
        if mp.server == 'pc' then xlua.progress(i, num_batches) end
        local batch = dataloader:sample_sequential_batch(false)
        local test_loss, prediction = model:fp(params_, batch)
        sum_loss = sum_loss + test_loss
    end
    local avg_loss = sum_loss/num_batches
    if mp.cuda then cutorch.synchronize() end
    collectgarbage()
    return avg_loss
end

function validate()
    local train_loss = test(train_test_loader, model.theta.params, false, math.min(5000, val_loader.num_batches))
    local val_loss = test(val_loader, model.theta.params, false, val_loader.num_batches)
    local test_loss = test(test_loader, model.theta.params, false, test_loader.num_batches)

    local log_string = 'train loss\t'..train_loss..
                      '\tval loss\t'..val_loss..
                      '\ttest_loss\t'..test_loss

    print(log_string)

    -- Save logs
    experimentLogger:add{['log MSE loss (train set)'] =  torch.log(train_loss),
                         ['log MSE loss (val set)'] =  torch.log(val_loss),
                         ['log MSE loss (test set)'] =  torch.log(test_loss)}
    experimentLogger:style{['log MSE loss (train set)'] = '~',
                           ['log MSE loss (val set)'] = '~',
                           ['log MSE loss (test set)'] = '~'}
   if mp.plot then experimentLogger:plot() end
    return train_loss, val_loss, test_loss
end

-- runs experiment
function experiment(start_iter, epoch_num)
    torch.setnumthreads(mp.num_threads)
    print('<torch> set nb of threads to ' .. torch.getnumthreads())
    train(start_iter, epoch_num)
end

function checkpoint(savefile, data, mp_)
    if mp_.cuda then
        data = data:float()
        torch.save(savefile, data)
        data = data:cuda()
    else
        torch.save(savefile, data)
    end
    collectgarbage()
end

function run_experiment()
    inittrain(false)
    experiment()
end


-- will have to change this soon
function read_log_file_3vals(logfile)
    local data1 = {}
    local data2 = {}
    local data3 = {}
    for line in io.lines(logfile) do
        local x = filter(function(x) return not(x=='') end,
                            stringx.split(line:gsub("%s+", ","),','))
        data1[#data1+1] = tonumber(x[1]) --ignores the string at the top
        data2[#data2+1] = tonumber(x[2]) --ignores the string at the top
        data3[#data3+1] = tonumber(x[3]) --ignores the string at the top
    end

    local data = torch.cat({torch.Tensor(data1), torch.Tensor(data2), torch.Tensor(data3)}, 2)

    -- test convergence
    local val = data[{{},{2}}]
    for w =3, data:size(1) do
        local valwin = torch.exp(val[{{-w,-1}}])
        local max_val_loss, max_val_loss_idx = torch.max(valwin,1)
        local min_val_loss, min_val_loss_idx = torch.min(valwin,1)
        local val_avg_delta = (valwin[{{2,-1}}] - valwin[{{1,-2}}]):mean()
        local abs_delta = (max_val_loss-min_val_loss):sum()
    end

    return data
end

-- UPDATE
function run_experiment_load()
    local snapshot = getLastSnapshot(mp.name)
    local snapshotfile = mp.savedir ..'/'..snapshot
    print(snapshotfile)
    local checkpoint = torch.load(snapshotfile)
    local saved_args = torch.load(mp.savedir..'/args.t7')
    mp = checkpoint.mp  -- completely overwrite  good
    mp.mode = 'expload'
    -- mp.cuda = true
    local iters = checkpoint.iters + 1

    train_losses = checkpoint.train_losses
    val_losses = checkpoint.val_losses
    test_losses = checkpoint.test_losses

    local logs_losses = read_log_file_3vals(mp.savedir..'/experiment.log')

    -- because previously we had not saved test_losses.
    if #test_losses == 0 then
        -- read it from the experiment log file
        test_losses = torch.exp(torch.squeeze(logs_losses[{{},{3}}])):totable()  -- TODO! you need to do torch exp here!
        assert(#test_losses==#train_losses)
    end


    if ((iters-1) >= mp.lrdecayafter and (iters-1) % mp.lrdecay_every == 0) then
        mp.lr = mp.lr*mp.lrdecay  -- because we usually decay right after we save?
    end
    optim_state = {learningRate   = mp.lr}
    print('Learning rate is now '..optim_state.learningRate)

    config_args = saved_args.config_args

    -- if mp.server == 'op' then mp.cuda = true end

    model_deps(mp.model)
    inittrain(true, mp.savedir ..'/'..snapshot, iters)  -- assuming the mp.savedir doesn't change

    -- you should now write the experiment logger
    -- you won't be able to write the train logger though! You'd have to save the original file
    assert(#train_losses==#val_losses and #train_losses==#test_losses)
    for i=1,#train_losses do
        local train_loss = train_losses[i]
        local val_loss = val_losses[i]
        local test_loss = test_losses[i]
        experimentLogger:add{['log MSE loss (train set)'] =  torch.log(train_loss),
                             ['log MSE loss (val set)'] =  torch.log(val_loss),
                             ['log MSE loss (test set)'] =  torch.log(test_loss)}
        experimentLogger:style{['log MSE loss (train set)'] = '~',
                               ['log MSE loss (val set)'] = '~',
                               ['log MSE loss (test set)'] = '~'}
    end

    -- These are things you have to set; although priority sampler might not be reset
    local epoch_num = math.floor(iters / train_loader.num_batches) + 1


    experiment(iters, epoch_num)
end

function model_deps(modeltype)
    if modeltype == 'lstmobj' or
            modeltype == 'ffobj' or
                    modeltype == 'gruobj' then
        M = require 'variable_obj_model'
    elseif modeltype == 'bffobj' then
        M = require 'branched_variable_obj_model'
    elseif modeltype == 'ind' then
        M = require 'independent'
    elseif modeltype == 'bl' then
        M = require 'blstm'
    elseif modeltype == 'np' then
        M = require 'nop'
    elseif modeltype == 'lstm' then
        M = require 'lstm'
    elseif modeltype == 'ff' then
        M = require 'feed_forward_model'
    else
        error('Unrecognized model')
    end
end

function getLastSnapshot(network_name)
    local res_file = io.popen("ls -t "..mp.logs_root..'/'..network_name..
                        " | grep -i epoch | head -n 1")
    local status, result = pcall(function()
        return res_file:read():match( "^%s*(.-)%s*$" ) end)
    print(result)
    res_file:close()
    if not status then return false else return result end
end

------------------------------------- Main -------------------------------------
if mp.mode == 'exp' then
    initsavebatches()
    print('Running experiment.')
    run_experiment()
elseif mp.mode == 'expload' then
    run_experiment_load()
elseif mp.mode == 'save' then
    initsavebatches()
else
    error('unknown mode')
end
