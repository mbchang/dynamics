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
cmd:option('-seed', true, 'manual seed or not')

-- dataset
cmd:option('-dataset_folders', '', 'dataset folder')
cmd:option('-test_dataset_folders', '', 'dataset folder')

-- model params
cmd:option('-rnn_dim', 50, 'hidden dimension')
cmd:option('-nbrhd', false, 'restrict attention to neighborhood')
cmd:option('-nbrhdsize', 4, 'number of radii out to look. nbhrdsize of 2 is when they exactly touching')
cmd:option('-layers', 3, 'layers in network')
cmd:option('-relative', true, 'relative state vs absolute state')
cmd:option('-diff', false, 'use relative context position and velocity state')
cmd:option('-accel', false, 'use acceleration data')
cmd:option('-batch_norm', false, 'batch norm')
cmd:option('-num_past', 2, 'number of past timesteps')


-- training options
cmd:option('-opt', "rmsprop", 'rmsprop | adam')
cmd:option('-batch_size', 50, 'batch size')
cmd:option('-shuffle', false, 'shuffle batches')
cmd:option('-max_iter', 3000000, 'max number of iterations (some huge number)')
cmd:option('-L2', 0, 'L2 regularization')  -- 0.001
cmd:option('-lr', 0.0003, 'learning rate')
cmd:option('-lrdecay', 0.99, 'learning rate annealing')
cmd:option('-val_window', 10, 'for testing convergence')
cmd:option('-val_eps', 1.5e-5, 'for testing convergence')  -- 1e-5
cmd:option('-im', false, 'infer mass')

-- priority sampling
cmd:option('-ps', true, 'turn on priority sampling')
cmd:option('-sharpen', 1, 'sharpen exponent')

-- experiment options
cmd:option('-plot', false, 'turn on/off plot')

-- every options
cmd:option('-print_every', 100, 'print every number of batches')
cmd:option('-save_every', 10000, 'save every number of batches')
cmd:option('-val_every',10000,'val every number of batches')
cmd:option('-lrdecay_every',2500,'decay lr every number of batches')
cmd:option('-lrdecayafter', 50000, 'number of epochs before turning down lr')

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
    mp.lrdecayafter = 50000
    mp.lrdecay_every = 2500
    mp.layers = 1
    mp.model = 'bffobj'
    mp.im = false
    mp.val_window = 5
    mp.val_eps = 2e-5
	mp.seq_length = 10  -- for the concatenate model
	mp.num_threads = 1
    mp.shuffle = false
    mp.print_every = 10
    mp.save_every = 10000
    mp.val_every = 20
    mp.plot = false--true
	mp.cuda = false
else
	-- mp.winsize = 3  -- total number of frames
    -- mp.num_past = 2 -- total number of past frames
    mp.num_future = 1
	mp.seq_length = 10   -- for the concatenate model
	mp.num_threads = 4
	mp.cuda = true
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
elseif mp.model == 'crnn' then 
    M = require 'clique_rnn'
elseif mp.model == 'lstmtime' then
    M = require 'lstm_model'
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

if mp.seed then torch.manualSeed(123) end
if mp.cuda then
    require 'cutorch'
    require 'cunn'
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
function inittrain(preload, model_path)
    print("Network parameters:")
    print(mp)
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
    print(model.network)
    print(model.theta.params:nElement(), 'parameters')

    trainLogger = optim.Logger(paths.concat(mp.savedir ..'/', 'train.log'))
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
    torch.save(mp.savedir..'/args.t7', {mp=mp,config_args=config_args})
    print("Initialized Network")
end

function initsavebatches()
    mp.cuda = false
    mp.cunn = false
    config_args.batch_size = mp.batch_size
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
end

-- closure: returns loss, grad_params
function feval_train(params_)  -- params_ should be first argument

    local batch = train_loader:sample_priority_batch(mp.sharpen)
    -- {
    --   1 : FloatTensor - size: 5x2x11
    --   2 : FloatTensor - size: 5x4x2x11
    --   3 : FloatTensor - size: 5x1x11
    --   4 : FloatTensor - size: 5x4x1x11
    --   5 : FloatTensor - size: 10
    -- }
    -- local batch = train_loader:sample_sequential_aggregated_batch(10, false)
    -- print(batch)
    -- assert(false)

    local loss, prediction = model:fp(params_, batch)
    local grad = model:bp(batch,prediction)

    -- L2 stuff

    if mp.L2 > 0 then
        -- Loss:
        loss = loss + mp.L2 * model.theta.params:norm(2)^2/2
        -- Gradients:
        model.theta.grad_params:add(model.theta.params:clone():mul(mp.L2) )
    end

    train_loader:update_batch_weight(loss)
    collectgarbage()
    return loss, grad -- f(x), df/dx
end

function train(start_iter, epoch_num)
    local epoch_num = epoch_num or 1
    local start_iter = start_iter or 1
    print('Start iter:', start_iter)
    print('Start epoch num:', epoch_num)
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
            v_train_loss, v_val_loss, v_tets_loss = validate()
            train_losses[#train_losses+1] = v_train_loss
            val_losses[#val_losses+1] = v_val_loss
            test_losses[#test_losses+1] = v_test_loss
            assert(mp.save_every % mp.val_every == 0 or
                    mp.val_every % mp.save_every == 0)

            -- save
            if (t-start_iter+1) % mp.save_every == 0 then
                local model_file = string.format('%s/epoch%.2f_%.4f.t7',
                                            mp.savedir, epoch_num, v_val_loss)
                print('saving checkpoint to ' .. model_file)
                model.network:clearState()

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

        if (t-start_iter+1) % train_loader.num_batches == 0 then
            epoch_num = t / train_loader.num_batches + 1
        end

        if mp.plot then trainLogger:plot() end
        if mp.cuda then cutorch.synchronize() end
        collectgarbage()
    end
end

function test(dataloader, params_, saveoutput)
    local sum_loss = 0
    for i = 1,dataloader.num_batches do
        if mp.server == 'pc' then xlua.progress(i, dataloader.num_batches) end
        local batch = dataloader:sample_sequential_batch(false)
        local test_loss, prediction = model:fp(params_, batch)
        sum_loss = sum_loss + test_loss
    end
    local avg_loss = sum_loss/dataloader.num_batches
    collectgarbage()
    return avg_loss
end

-- a table of onehot tensors of size num_hypotheses
function generate_onehot_hypotheses(num_hypotheses)
    local hypotheses = {}
    for i=1,num_hypotheses do
        local hypothesis = torch.zeros(num_hypotheses)
        hypothesis[{{i}}]:fill(1)
        table.insert(hypotheses, hypothesis)
    end
    return hypotheses
end

function infer_properties(dataloader, params_, property, method)
    -- TODO for other properties
    local hypotheses, si_indices, num_hypotheses
    if property == 'mass' then
        si_indices = tablex.deepcopy(config_args.si.m)
        si_indices[2] = si_indices[2]-1  -- ignore mass 1e30
        num_hypotheses = si_indices[2]-si_indices[1]+1
        hypotheses = generate_onehot_hypotheses(num_hypotheses) -- good
    elseif property == 'size' then 
        assert(false, property..'not implemented')
    end

    local accuracy
    if method == 'backprop' then 
        accuracy = backprop2input(dataloader, params_, hypotheses, si_indices)
    elseif method == 'max_likelihood' then
        accuracy = max_likelihood(dataloader, params_, hypotheses, si_indices)
    end
    return accuracy
end

-- copies batch
function apply_hypothesis(batch, hyp, si_indices)
    local this_past, context_past, this_future, context_future, mask = unpack(batch)
    this_past = this_past:clone()
    context_past = context_past:clone()
    this_future = this_future:clone()
    context_future = context_future:clone()

    -- I should apply the hypothesis to the entire batch?
    -- later I will have to compare to the original to measure accuracy.
    -- well since I'm going to go through all hypotheses anyways it don't have
    -- to worry about things within the batch. But I have to save the original
    -- this_past for ground truth so that I can compare and measure how many 
    -- within the batch, after I applied all my hypotheses, had the best error.
    local num_ex = this_past:size(1)
    local num_context = this_past:size(2)
    this_past[{{},{},si_indices}] = torch.repeatTensor(hyp, num_ex, num_context, 1)
    return {this_past, context_past, this_future, context_future, mask}
end


function backprop2input(dataloader, params_, si_indices)
    local num_correct = 0
    local count = 0
    local batch_group_size = 1000
    for i = 1, dataloader.num_batches, batch_group_size do
        if mp.server == 'pc' then xlua.progress(i, dataloader.num_batches) end
        
        local batch = dataloader:sample_sequential_batch(false)

        -- return best_hypotheses by backpropagating to the input
        --  initial hypothesis should be 0.5 in all the indices
        local hypothesis_length = si_indices[2]-si_indices[1]+1
        local initial_hypothesis = torch.Tensor(hypothesis_length):fill(0.5)
        local hypothesis_batch = apply_hypothesis(batch, initial_hypothesis, si_indices)  -- good

        local this_past_orig, context_past_orig, this_future_orig, context_future_orig, mask_orig = unpack(hypothesis_batch)

        -- TODO: keep track of whether it is okay to let this_past, context_past, this_future, context_future, mask get mutated!
        -- closure: returns loss, grad_Input
        -- out of scope: batch, si_indices
        function feval_b2i(this_past_hypothesis)
            -- build modified batch
            local updated_hypothesis_batch = {this_past_hypothesis, context_past_orig:clone(), this_future_orig:clone(), context_future_orig:clone(), mask_orig:clone()}

            model.network:clearState() -- zeros out gradInput

            local loss, prediction = model:fp(params_, updated_hypothesis_batch)
            local d_input = model:bp_input(updated_hypothesis_batch,prediction)

            -- 0. Get names
            local d_pairwise = d_input[1]
            local d_identity = d_input[2]

            -- 1. assert that all the d_inputs in pairwise are equal
            local d_focus_in_pairwise = {}
            for i=1,#d_pairwise do
                table.insert(d_focus_in_pairwise, d_pairwise[i][1])
            end
            assert(alleq_tensortable(d_focus_in_pairwise))

            -- 2. Get the gradients that you need to add
            local d_pairwise_focus = d_pairwise[1][1]:clone()  -- pick first one
            local d_identity_focus = d_identity:clone()
            assert(d_pairwise_focus:isSameSizeAs(d_identity_focus))

            -- 3. Add the gradients
            local d_focus = d_pairwise_focus + d_identity_focus

            -- 4. Zero out everything except the property that you are performing inference on
            d_focus:resize(mp.batch_size, mp.num_past, mp.object_dim)
            if si_indices[1] > 1 then
                d_focus[{{},{},{1,si_indices[1]-1}}]:zero()
            end
            if si_indices[2] < mp.object_dim then
                d_focus[{{},{},{1,si_indices[2]+1}}]:zero()
            end
            d_focus:resize(mp.batch_size, mp.num_past*mp.object_dim)

            -- 6. Check that weights have not been changed
                -- check that all the gradParams are 0 in the network
            assert(model.theta.grad_params:norm() == 0)

            collectgarbage()
            return loss, d_focus -- f(x), df/dx
        end

        local num_iters = 10 -- TODO: change this (perhaps you can check for convergence. you just need rough convergence)
        local b2i_optstate = {learningRate = 0.01}  -- TODO tune this 
        local this_past_hypothesis = this_past_orig:clone()

        -- get old model parameters

        for t=1,num_iters do

            local new_this_past_hypothesis, train_loss = optimizer(feval_b2i,
                                this_past_hypothesis, b2i_optstate)  -- next batch

            -- 1. Check that the model parameters have not changed
            assert(model.theta.parameters:equal(old_model_parameters))

            -- 2. Check that the input focus object has changed 
                -- do this outside of feval
                -- very unlikely they are equal, unless gradient was 0
            assert(not(this_past_hypothesis:equal(new_this_past_hypothesis)))

            -- 3. update this_past
            this_past_hypothesis = new_this_past_hypothesis  -- TODO: can just update it above
            collectgarbage()
        end

        -- now you have a this_past as your hypothesis. Select and binarize.
        -- TODO check that this properly gets mutated
        this_past_hypothesis[{{},{},si_indices}] = binarize(this_past_hypothesis[{{},{},si_indices}]) -- NOTE you are assuming the num_future is 1

        -- now that you have best_hypothesis, compare best_hypotheses with truth
        -- need to construct true hypotheses based on this_past, hypotheses as parameters
        local ground_truth = torch.squeeze(this_past_orig[{{},{-1},si_indices}])  -- object properties always the same across time
        local num_equal = ground_truth:eq(this_past_hypothesis[{{},{},si_indices}]):sum(2):eq(hypothesis_length):sum()
        num_correct = num_correct + num_equal
        count = count + mp.batch_size
        collectgarbage()
    end
    return num_correct/count
end

-- selects max over each row in last axis
-- makes the max one and everything else 0
function binarize(tensor)
    local y, i = torch.max(tensor, tensor:dim())
    tensor:zero()
    tensor:indexFill(tensor:dim(), torch.squeeze(i,tensor:dim()), 1)
    return tensor
end


function max_likelihood(dataloader, params_, hypotheses, si_indices)
    local num_correct = 0
    local count = 0
    for i = 1, dataloader.num_batches do
        if mp.server == 'pc' then xlua.progress(i, dataloader.num_batches) end
        local batch = dataloader:sample_sequential_batch(false)
        local best_losses = torch.Tensor(mp.batch_size):fill(math.huge)
        local best_hypotheses = torch.zeros(mp.batch_size,#hypotheses)

        for j,h in pairs(hypotheses) do
            local hypothesis_batch = apply_hypothesis(batch, h, si_indices)  -- good
            local test_losses, prediction = model:fp_batch(params_, hypothesis_batch)  -- good

            -- test_loss is a tensor of size bsize
            local update_indices = test_losses:lt(best_losses):nonzero()

            if update_indices:nElement() > 0 then
                update_indices = torch.squeeze(update_indices,2)
                --best_loss should equal test loss at the indices where test loss < best_loss
                best_losses:indexCopy(1,update_indices,test_losses:index(1,update_indices))

                -- best_hypotheses should equal h at the indices where test loss < best_loss
                best_hypotheses:indexCopy(1,update_indices,torch.repeatTensor(h,update_indices:size(1),1))
            end
            -- check that everything has been updated
            assert(not(best_losses:equal(torch.Tensor(mp.batch_size):fill(math.huge))))
            assert(not(best_hypotheses:equal(torch.zeros(mp.batch_size,#hypotheses))))
        end

        -- now that you have best_hypothesis, compare best_hypotheses with truth
        -- need to construct true hypotheses based on this_past, hypotheses as parameters
        local this_past = batch[1]:clone()
        local ground_truth = torch.squeeze(this_past[{{},{-1},si_indices}])  -- object properties always the same across time
        local num_equal = ground_truth:eq(best_hypotheses):sum(2):eq(#hypotheses):sum()
        num_correct = num_correct + num_equal
        count = count + mp.batch_size
        collectgarbage()
    end
    return num_correct/count
end


function validate()
    local train_loss = test(train_test_loader, model.theta.params, false)
    local val_loss = test(val_loader, model.theta.params, false)
    local test_loss = test(test_loader, model.theta.params, false)

    -- local train_loss = 0
    -- local test_loss = 0



    local log_string = 'train loss\t'..train_loss..
                      '\tval loss\t'..val_loss..
                      '\ttest_loss\t'..test_loss

    if mp.im then
        local mass_accuracy = infer_properties(val_loader, model.theta.params, 'mass', 'max_likelihood')
        log_string = log_string..'\tmass accuracy\t'..mass_accuracy
        inferenceLogger:add{['Mass accuracy (val set)'] = mass_accuracy}
        inferenceLogger:style{['Mass accuracy (val set)'] = '~'}
    end

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


-- UPDATE
function run_experiment_load()
    local snapshot = getLastSnapshot(mp.name)
    print(snapshot)
    -- local checkpoint = torch.load(mp.savedir ..'/'..snapshot)
-- hardcoded
    local checkpoint = torch.load('logs/balls_n3_t60_ex20,balls_n6_t60_ex20,balls_n5_t60_ex20/epoch1.00_0.0469.t7')

    mp = checkpoint.mp  -- completely overwrite  NOTE!

    print(mp.lr)

    assert(false)
    inittrain(true, mp.savedir ..'/'..snapshot)  -- assuming the mp.savedir doesn't change

    -- These are things you have to set; although priority sampler might not be reset
    local iters = mp.val_every * #checkpoint.val_losses + 1
    -- local epoch_num = math.floor(iters / train_loader.num_batches) + 1
    local epoch_num = 1

    mp.lr = 1.077384359378e-05
    optim_state = {learningRate   = mp.lr}

    experiment(iters, epoch_num)
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
