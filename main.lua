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
local D2 = require 'datasaver'
require 'logging_utils'

-- torch.setdefaulttensortype('torch.FloatTensor')

------------------------------------- Init -------------------------------------
-- Best val: 1/29/16: baselinesubsampledcontigdense_opt_adam_testcfgs_[:-2:2-:]_traincfgs_[:-2:2-:]_lr_0.005_batch_size_260.out

mp = lapp[[
   -e,--mode          (default "exp")           exp | pred
   -d,--root          (default "logslink")      	subdirectory to save logs
   -m,--model         (default "ff")   		type of model tor train: lstm | ff
   -n,--name          (default "save3")
   -p,--plot          (default true)                    	plot while training
   -j,--traincfgs     (default "[:-2:2-:]")
   -k,--testcfgs      (default "[:-2:2-:]")
   -b,--batch_size    (default 60)
   -l,--accel         (default true)
   -o,--opt           (default "optimrmsprop")       rmsprop | adam | optimrmsprop
   -c,--server		  (default "op")			pc=personal | op = openmind
   -t,--relative      (default "true")           relative state vs abs state
   -s,--shuffle  	  (default "false")
   -r,--lr            (default 0.005)      	   learning rate
   -a,--lrdecay       (default 0.9)            annealing rate
   -h,--sharpen       (default 1)               sharpen exponent
   -f,--lrdecayafter  (default 50)              number of epochs before turning down lr
   -i,--max_epochs    (default 1000)           	maximum nb of iterations per batch, for LBFGS
   --diff             (default "true")
   --rnn_dim          (default 50)
   --layers           (default 1)
   --seed             (default "true")
   --max_grad_norm    (default 10)
   --save_output	  (default false)
   --print_every      (default 100)
   --save_every       (default 1)
]]

if mp.server == 'pc' then
    mp.root = 'logs'
    mp.winsize = 20  -- total number of frames
    mp.num_past = 10 --10
    mp.num_future = 10 --10
	mp.dataset_folder = '/Users/MichaelChang/Documents/Researchlink/SuperUROP/Code/opdata/7'--dataset_files_subsampled_dense_np2' --'hoho'
    mp.traincfgs = '[:-2:2-:]'
    mp.testcfgs = '[:-2:2-:]'
	mp.batch_size = 30 --1
    mp.lrdecay = 1
	mp.seq_length = 10
	mp.num_threads = 1
    mp.print_every = 1
    mp.plot = false--true
	mp.cuda = false
	mp.cunn = false
    -- mp.max_epochs = 5
else
	mp.winsize = 2  -- total number of frames
    mp.num_past = 1 -- total number of past frames
    mp.num_future = 1
	mp.dataset_folder = '/om/data/public/mbchang/physics-data/6'
	mp.seq_length = 10
	mp.num_threads = 4
    mp.plot = false
	mp.cuda = true
	mp.cunn = true
end

local M
if mp.model == 'lstm' then
    M = require 'model_new'
elseif mp.model == 'ff' then
    M = require 'feed_forward_model'
else
    error('Unrecognized model')
end

mp.object_dim = 8.0  -- hardcoded
if mp.accel then mp.object_dim = 10 end
mp.input_dim = mp.object_dim*mp.num_past
mp.out_dim = mp.object_dim*mp.num_future
mp.savedir = mp.root .. '/' .. mp.name

if mp.seed then torch.manualSeed(123) end
if mp.shuffle == 'false' then mp.shuffle = false end
if mp.relative == 'false' then mp.relative = false end
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
                              mp.shuffle,
                              mp.cuda}
    train_loader = D.create('trainset', unpack(data_loader_args))
    val_loader =  D.create('valset', unpack(data_loader_args))  -- using testcfgs
    test_loader = D.create('testset', unpack(data_loader_args))
    train_test_loader = D.create('trainset', unpack(data_loader_args))
    model = M.create(mp, preload, model_path)

    trainLogger = optim.Logger(paths.concat(mp.savedir ..'/', 'train.log'))
    experimentLogger = optim.Logger(paths.concat(mp.savedir ..'/', 'experiment.log'))
    if mp.plot == false then
        trainLogger.showPlot = false
        experimentLogger.showPlot = false
    end
    print("Initialized Network")
end

-- function initsavebatches(preload, model_path)
--     mp.cuda = false
--     mp.cunn = false
--     mp.shuffle = false
--     print("Network parameters:")
--     print(mp)
--     local data_loader_args = {mp.dataset_folder,
--                               mp.batch_size,
--                               mp.shuffle,
--                               mp.cuda,
--                               mp.relative,
--                               mp.num_past,
--                               mp.winsize}
--     train_loader = D.create('trainset', D.convert2allconfigs(mp.traincfgs), unpack(data_loader_args))
--     val_loader =  D.create('valset', D.convert2allconfigs(mp.testcfgs), unpack(data_loader_args))  -- using testcfgs
--     test_loader = D.create('testset', D.convert2allconfigs(mp.testcfgs), unpack(data_loader_args))
--
--     train_loader:save_sequential_batches()
--     val_loader:save_sequential_batches()
--     test_loader:save_sequential_batches()
-- end

function initsavebatches(preload, model_path)
    mp.cuda = false
    mp.cunn = false
    mp.shuffle = false
    print("Network parameters:")
    print(mp)
    local data_loader_args = {mp.dataset_folder,
                              mp.batch_size,
                              mp.shuffle,
                              mp.relative,
                              mp.num_past,
                              mp.winsize}
    train_loader = D2.create('trainset', mp.traincfgs, unpack(data_loader_args))
    val_loader =  D2.create('valset', mp.testcfgs, unpack(data_loader_args))  -- using testcfgs
    test_loader = D2.create('testset', mp.testcfgs, unpack(data_loader_args))

    train_loader:save_sequential_batches()
    val_loader:save_sequential_batches()
    test_loader:save_sequential_batches()
end

function inittest(preload, model_path)
    print("Network parameters:")
    print(mp)
    local data_loader_args = {mp.dataset_folder,
                              mp.shuffle,
                              mp.cuda}
    test_loader = D.create('testset', unpack(data_loader_args))  -- TODO: Testing on trainset
    model = M.create(mp, preload, model_path)
    if preload then mp = torch.load(model_path).mp end
    modelfile = model_path
    print("Initialized Network")
end



-- closure: returns loss, grad_params
function feval_train(params_)  -- params_ should be first argument

    local this, context, y, mask = unpack(train_loader:sample_priority_batch(mp.sharpen))
    y = crop_future(y, {y:size(1), mp.winsize-mp.num_past, mp.object_dim}, {2,mp.num_future})   -- TODO RESIZE THIS

    local loss, _ = model:fp(params_, {this=this,context=context}, y, mask)
    local grad = model:bp({this=this,context=context}, y, mask)
    train_loader.priority_sampler:update_batch_weight(train_loader.current_sampled_id, loss)
    collectgarbage()
    return loss, grad -- f(x), df/dx
end

-- tensor (batchsize, winsize*obj_dim)
-- reshapesize (batchsize, winsize, obj_dim)
-- cropdim (dim, amount_to_take) == (dim, mp.num_future)
function crop_future(tensor, reshapesize, cropdim)
    local crop = tensor:clone()
    crop = crop:reshape(unpack(reshapesize))
    --hacky
    if crop:dim() == 3 then
        assert(cropdim[1]==2)
        crop = crop[{{},{1,cropdim[2]},{}}]  -- (num_samples x num_future x 8) -- TODO the -1 should be a function of 1+num_future
        crop = crop:reshape(reshapesize[1], cropdim[2] * mp.object_dim)
    else
        assert(crop:dim()==4 and cropdim[1] == 3)
        crop = crop[{{},{},{1,cropdim[2]},{}}]
        crop = crop:reshape(reshapesize[1], mp.seq_length, cropdim[2] * mp.object_dim)   -- TODO RESIZE THIS (use reshape size here)
    end
    return crop
end

-- trains for one epoch
function train(epoch_num)
    local new_params, train_loss
    local loss_run_avg = 0
    for t = 1,train_loader.num_batches do
        train_loader.priority_sampler:set_epcnum(epoch_num)--set_epcnum
        -- xlua.progress(t, train_loader.num_batches)
        new_params, train_loss = optimizer(feval_train, model.theta.params, optim_state)  -- next batch
        assert(new_params == model.theta.params)
        if t % mp.print_every == 0 then
            print(string.format("epoch %2d\titeration %2d\tloss = %6.8f\tgradnorm = %6.4e\tbatch = %4d\thardest batch: %4d \twith loss %6.8f lr = %6.4e",
                    epoch_num, t, train_loss[1],
                    model.theta.grad_params:norm(),
                    train_loader.current_sampled_id,
                    train_loader.priority_sampler:get_hardest_batch()[2],
                    train_loader.priority_sampler:get_hardest_batch()[1],
                    optim_state.learningRate))
        end
        loss_run_avg = loss_run_avg + train_loss[1]
        trainLogger:add{['log MSE loss (train set)'] =  torch.log(train_loss[1])}
        trainLogger:style{['log MSE loss (train set)'] = '~'}
        if mp.plot then trainLogger:plot() end
        if mp.cuda then cutorch.synchronize() end
        collectgarbage()
    end
    return loss_run_avg/train_loader.num_batches -- running avg of training loss.
end

-- test on dataset
function test(dataloader, params_, saveoutput)
    local sum_loss = 0
    for i = 1,dataloader.num_batches do
        if mp.server == 'pc ' then xlua.progress(i, dataloader.num_batches) end

        -- get batch
        local this, context, y, mask, config, start, finish, context_future = unpack(dataloader:sample_sequential_batch())

        -- if mp.cuda then
        --     this = this:cuda()
        --     context = context:cuda()
        --     y = y:cuda()
        --     context_future = context_future:cuda()
        -- end

        y = crop_future(y, {y:size(1), mp.winsize-mp.num_past, mp.object_dim}, {2,mp.num_future})  -- TODO RESIZE THIS
        context_future = crop_future(context_future, {context_future:size(1), mp.seq_length, mp.winsize-mp.num_past, mp.object_dim}, {3,mp.num_future})

        -- predict
        local test_loss, prediction = model:fp(params_, {this=this,context=context}, y, mask)

        -- reshape to -- (num_samples x num_future x 8)
        prediction = prediction:reshape(mp.batch_size, mp.num_future, mp.object_dim)   -- TODO RESIZE THIS
        this = this:reshape(mp.batch_size, mp.num_past, mp.object_dim)   -- TODO RESIZE THIS
        y = y:reshape(mp.batch_size, mp.num_future, mp.object_dim)   -- TODO RESIZE THIS

        -- take care of relative position
        if mp.relative then
            prediction[{{},{},{1,4}}] = prediction[{{},{},{1,4}}] + this[{{},{-1},{1,4}}]:expandAs(prediction[{{},{},{1,4}}])  -- TODO RESIZE THIS?
            y[{{},{},{1,4}}] = y[{{},{},{1,4}}] + this[{{},{-1},{1,4}}]:expandAs(y[{{},{},{1,4}}]) -- add back because relative  -- TODO RESIZE THIS?

            -- prediction = prediction + this[{{},{-1}}]:expandAs(prediction)  -- TODO RESIZE THIS?
            -- y = y - this[{{},{-1}}]:expandAs(y) -- add back because relative  -- TODO RESIZE THIS?
        end

        -- save
        if saveoutput then
            save_example_prediction({this, context, y, prediction, context_future},
                                    {config, start, finish},
                                    modelfile,
                                    dataloader,
                                    mp.num_future)
        end
        sum_loss = sum_loss + test_loss
    end
    local avg_loss = sum_loss/dataloader.num_batches
    collectgarbage()
    return avg_loss
end

-- test on dataset
-- assumption: that you predict 1 out
--[[
    Want:
        the data given to still have windowsize of 20.
        But we want to only predict for 1 timestep,
        But we want the context to have full windowsize. Perhaps we can
            have a flag to set? The full context, starting at the time prediction
            starts until the end of the window, should be passed through
        But also you need the full context and context_future to be passed in.
            Where to do the processing?
        Given the pred (batch_size, num_future, obj_dim) we will take the
        first time slice: [{{},{1}}] of that

        -- TODO wait, do we want dataloader.num_future to be constrained to be 1 at this point?
-- ]]
--
function simulate(dataloader, params_, saveoutput, numsteps)
    assert(mp.num_future == 1 and numsteps <= mp.winsize-mp.num_past)
    local sum_loss = 0
    for i = 1,dataloader.num_batches do
        if mp.server == 'pc ' then xlua.progress(i, dataloader.num_batches) end

        -- get data
        local this_orig, context_orig, y_orig, mask, config, start, finish, context_future_orig = unpack(dataloader:sample_sequential_batch())
        local this, context = this_orig:clone(), context_orig:clone()
        context_future = context_future_orig:clone():reshape(mp.batch_size, mp.seq_length, mp.winsize-mp.num_past, mp.object_dim)

        -- at this point:
        -- this (bsize, numpast*objdim)  -- TODO RESIZE THIS
        -- context (bsize, mp.seq_length, mp.numpast*mp.objdim)
        -- y_orig (bsize, (mp.winsize-mp.num_past)*objdim)
        -- context_future (bsize, mp.seqlength, (mp.winsize-mp.num_past)*mp.objdim)

        -- allocate space, already assume reshape
        local pred_sim = model_utils.transfer_data(torch.zeros(mp.batch_size, numsteps, mp.object_dim), mp.cuda)  -- TODO RESIZE THIS
        for t = 1,numsteps do
            -- the t-th timestep in the future
            y = y_orig:clone():reshape(mp.batch_size, mp.winsize-mp.num_past, mp.object_dim)[{{},{t},{}}]  -- increment time in y; may need to reshape  -- TODO RESIZE THIS
            y = y:reshape(mp.batch_size, 1*mp.object_dim)  -- TODO RESIZE THIS

            local test_loss, prediction = model:fp(params_, {this=this,context=context}, y, mask)  -- TODO Does mask make sense here? Well, yes right? because mask only has to do with the objects

            prediction = prediction:reshape(mp.batch_size, mp.num_future, mp.object_dim)  -- TODO RESIZE THIS
            this = this:reshape(mp.batch_size, mp.num_past, mp.object_dim)  -- TODO RESIZE THIS
            context = context:reshape(mp.batch_size, mp.seq_length, mp.num_past, mp.object_dim)

            if mp.relative then
                prediction = prediction + this[{{},{-1}}]:expandAs(prediction) -- should this be ground truth? During test time no.  -- TODO RESIZE THIS
            end

            -- update this and context
            -- chop off first time step and then add in next one
            if mp.num_past > 1 then
                this = torch.cat({this[{{},{2,-1},{}}]:clone(), prediction}, 2)  -- you can add y in here  -- TODO RESIZE THIS
                context = torch.cat({context[{{},{},{2,-1},{}}]:clone(), context_future[{{},{},{t},{}}]:clone()},3)
            else
                assert(mp.num_past == 1)
                this = prediction:clone()  -- just use prev prediction as input
                context = context_future[{{},{},{t},{}}]:clone()
            end

            -- reshape it back
            this = this:reshape(mp.batch_size, mp.num_past*mp.object_dim)  -- TODO RESIZE THIS
            context = context:reshape(mp.batch_size, mp.seq_length, mp.num_past*mp.object_dim)

            pred_sim[{{},{t},{}}] = prediction  -- note that this is just one timestep  -- you can add y in here
            sum_loss = sum_loss + test_loss
        end

        -- reshape to -- (num_samples x num_future x 8)
        this_orig = this_orig:reshape(this_orig:size(1), mp.num_past, mp.object_dim)  -- will render the original past  -- TODO RESIZE THIS
        y_orig = y_orig:reshape(y_orig:size(1), mp.winsize-mp.num_past, mp.object_dim) -- will render the original future  -- TODO RESIZE THIS
        context_future_orig = context_future_orig:reshape(mp.batch_size, mp.seq_length, mp.winsize-mp.num_past, mp.object_dim)

        if mp.relative then
            y_orig = y_orig + this_orig[{{},{-1}}]:expandAs(y_orig)  -- TODO RESIZE THIS
        end

        -- now crop only the number of timesteps you need; pred_sim is also this many timesteps
        y_orig = y_orig[{{},{1,numsteps},{}}]  -- TODO RESIZE THIS
        context_future_orig = context_future_orig[{{},{},{1,numsteps},{}}]

        if saveoutput then
            save_example_prediction({this_orig, context_orig, y_orig, pred_sim, context_future_orig},
                                    {config, start, finish},
                                    modelfile,
                                    dataloader,
                                    numsteps)
        end
    end
    local avg_loss = sum_loss/dataloader.num_batches/numsteps
    collectgarbage()
    return avg_loss
end

function save_example_prediction(example, description, modelfile_, dataloader, numsteps)
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

    -- For now, just save it as hdf5. You can feed it back in later if you'd like
    save_to_hdf5(save_path,
        {pred=prediction,
        this=this:reshape(this:size(1), -- TODO RESIZE THIS
                    mp.num_past,
                    mp.object_dim),
        context=context:reshape(context:size(1),
                    context:size(2),
                    mp.num_past,
                    mp.object_dim),
        y=y:reshape(y:size(1),  -- TODO RESIZE THIS (Probably reshaping y is unnecessary)
                    numsteps,
                    mp.object_dim),
        context_future=context_future:reshape(context_future:size(1),
                    context_future:size(2),
                    numsteps,
                    mp.object_dim)})
end

-- runs experiment
function experiment()
    torch.setnumthreads(mp.num_threads)
    print('<torch> set nb of threads to ' .. torch.getnumthreads())
    local train_losses, val_losses, test_losses = {},{},{}
    for i = 1, mp.max_epochs do
        print('Learning rate is now '..optim_state.learningRate)
        train(i)
        local train_loss = test(train_test_loader, model.theta.params, false)
        local val_loss = test(val_loader, model.theta.params, false)
        local test_loss = test(test_loader, model.theta.params, false)
        print('train loss\t'..train_loss..'\tval loss\t'..val_loss..'\ttest_loss\t'..test_loss)

        -- Save logs
        experimentLogger:add{['log MSE loss (train set)'] =  torch.log(train_loss),
                             ['log MSE loss (val set)'] =  torch.log(val_loss),
                             ['log MSE loss (test set)'] =  torch.log(test_loss)}
        experimentLogger:style{['log MSE loss (train set)'] = '~',
                               ['log MSE loss (val set)'] = '~',
                               ['log MSE loss (test set)'] = '~'}
        train_losses[#train_losses+1] = train_loss
        val_losses[#val_losses+1] = val_loss
        test_losses[#test_losses+1] = test_loss

        if i % mp.save_every == 0 then
            -- checkpoint(mp.savedir .. '/network'..'epc'..i..'.t7', model.network, mp) -- model.rnns[1]?
            -- checkpoint(mp.savedir .. '/params'..'epc'..i..'.t7', model.theta.params, mp)

            local model_file = string.format('%s/epoch%.2f_%.4f.t7', mp.savedir, i, val_loss)
            print('saving checkpoint to ' .. model_file)
            local checkpoint = {}
            checkpoint.model = model
            checkpoint.mp = mp
            checkpoint.train_losses = train_losses
            checkpoint.val_losses = val_losses
            torch.save(model_file, checkpoint)

            print('Saved model')
        end
        if mp.plot then experimentLogger:plot() end
        if mp.cuda then cutorch.synchronize() end

        -- here you can adjust the learning rate based on val loss
        if i >= mp.lrdecayafter then
            optim_state.learningRate =optim_state.learningRate*mp.lrdecay
        end
        collectgarbage()
    end
end

function checkpoint(savefile, data, mp_)
    if mp_.cuda then
        -- data = data:float()
        torch.save(savefile, data)
        -- data = data:cuda()
    else
        torch.save(savefile, data)
    end
    collectgarbage()
end

function run_experiment()
    inittrain(false)
    experiment()
end


function getLastSnapshot(network_name)
    local res_file = io.popen("ls -t "..mp.root..'/'..network_name.." | grep -i epoch | head -n 1")
    local status, result = pcall(function() return res_file:read():match( "^%s*(.-)%s*$" ) end)
    res_file:close()
    if not status then
        return false
    else
        return result
    end
end

function predict()
    local snapshot = getLastSnapshot('save3')
    print(snapshot)
    -- assert(false)
    mp = torch.load(mp.savedir ..'/'..snapshot).mp
    inittest(true, mp.savedir ..'/'..snapshot)  -- assuming the mp.savedir doesn't change

    -- inittest(true, mp.savedir ..'/'..'network.t7')  -- assuming the mp.savedir doesn't change
    print(test(test_loader, torch.load(mp.savedir..'/'..'params.t7'), true))
end

function predict_simulate()
    inittest(true, mp.savedir ..'/'..'network.t7')
    print(simulate(test_loader, torch.load(mp.savedir..'/'..'params.t7'), true, 5))
end


-- ------------------------------------- Main -------------------------------------
if mp.mode == 'exp' then
    run_experiment()
elseif mp.mode == 'sim' then
    predict_simulate()
elseif mp.mode == 'save' then
    initsavebatches(false)
else
    predict()
end


-- inittest(false, mp.savedir ..'/'..'network.t7')
-- print(simulate(test_loader, model.theta.params, false, 3))
