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
local pls = require 'pl.stringx'

-- require 'nn'
require 'rnn'  -- also installs moses (https://github.com/Yonaba/Moses/blob/master/doc/tutorial.md), like underscore.js
require 'torch'
require 'nngraph'
require 'Base'
require 'IdentityCriterion'
require 'gnuplot'

-- Local Imports
local model_utils = require 'model_utils'
-- local D = require 'data_sampler'
local D = require 'general_data_sampler'
local D2 = require 'datasaver'
require 'logging_utils'
require 'json_interface'

-- hacky for now, just to see if it works
local data_process = require 'data_process'

------------------------------------- Init -------------------------------------
local cmd = torch.CmdLine()
cmd:option('-mode', "exp", 'exp | pred | simulate | save')
cmd:option('-server', "op", 'pc = personal | op = openmind')
cmd:option('logs_root', 'logs', 'subdirectory to save logs and checkpoints')
cmd:option('data_root', '../data', 'subdirectory to save data')
cmd:option('-name', "mj", 'experiment name')
cmd:option('-seed', true, 'manual seed or not')
-- dataset
cmd:option('-test_dataset_folders', '', 'dataset folder')
-- experiment options
-- cmd:option('-gt', false, 'saving ground truth')  -- 0.001
cmd:option('-ns', 3, 'number of test batches')
cmd:option('-steps', 1, 'steps to simulate')
cmd:text()

-- parse input params
mp = cmd:parse(arg)

if mp.server == 'pc' then
    mp.data_root = 'mj_data'
    mp.logs_root = 'logs'
    mp.winsize = 10 -- total number of frames
    mp.num_past = 2 --10
    mp.num_future = 1 --10
	mp.batch_size = 5 --1
	mp.seq_length = 10
	mp.num_threads = 1
	mp.cuda = false
else
	mp.winsize = 10  -- total number of frames
    mp.num_past = 2 -- total number of past frames
    mp.num_future = 1
	mp.seq_length = 10
	mp.num_threads = 4
	mp.cuda = true
end

local M

-- world constants
local subsamp = 1

-- mp.input_dim = mp.object_dim*mp.num_past
-- mp.out_dim = mp.object_dim*mp.num_future
mp.name = string.gsub(string.gsub(string.gsub(mp.name,'{',''),'}',''),"'",'')
mp.test_dataset_folders = assert(loadstring("return "..string.gsub(mp.test_dataset_folders,'\"',''))())
mp.savedir = mp.logs_root .. '/' .. mp.name
mp.relative=true -- TODO: address this!

if mp.seed then torch.manualSeed(123) end
if mp.cuda then
    require 'cutorch'
    require 'cunn'
end

local model, test_loader, modelfile, dp

------------------------------- Helper Functions -------------------------------

function inittest(preload, model_path, opt)
    print("Network parameters:")
    print(mp.test_dataset_folders)
    dp = data_process.create(jsonfile, outfile, config_args)  -- TODO: actually you might want to load configs, hmmm so does eval need jsonfile, outfile?
    model = M.create(mp, preload, model_path)
    mp.cuda = false -- NOTE HACKY
    local data_loader_args = {data_root=mp.data_root..'/',
                              dataset_folders=mp.test_dataset_folders,
                              maxwinsize=config_args.maxwinsize,
                              winsize=mp.winsize, -- not sure if this should be in mp
                              num_past=mp.num_past,
                              num_future=mp.num_future,
                              relative=mp.relative,
                              sim=opt.sim,
                              cuda=mp.cuda
                            }
    test_loader = D.create('testset', tablex.deepcopy(data_loader_args))  -- TODO: Testing on trainset
    modelfile = model_path
    print("Initialized Network")
end

function backprop2input()
    -- for one input

    -- get batch
    local batch = test_loader:sample_sequential_batch()  -- TODO replace with some other data loader!
    -- NOTE CHANGE BATCH HERE
    local this, context, y, mask = unpack(batch)
    local x = {this=this,context=context}

    if not sim then
        y = crop_future(y, {y:size(1), mp.winsize-mp.num_past, mp.object_dim},
                            {2,mp.num_future})
    end
    -- convert_type

    -- unpack inputs
    local this_past     = convert_type(x.this:clone(), mp.cuda)
    local context       = convert_type(x.context:clone(), mp.cuda)
    local this_future   = convert_type(y:clone(), mp.cuda)

    function feval_back2mass(inp)
        -- inp is this_past
        -- forward
        local splitter = split_output(mp)
        local preproc = preprocess_input(mask)
        if mp.cuda then preproc:cuda() end
        local input = preproc:forward{inp,context}  -- this changes, context doesn't
        if torch.find(mask,1)[1] == 1 then input = {input} end

        local prediction = model.network:forward(input)
        local p_pos, p_vel, p_obj_prop = unpack(splitter:forward(prediction))
        local gt_pos, gt_vel, gt_obj_prop =
                            unpack(split_output(mp):forward(this_future))
        local loss = model.criterion:forward(p_vel, gt_vel)

        -- backward
        local d_pos = model.identitycriterion:backward(p_pos, gt_pos)
        local d_vel = model.criterion:backward(p_vel, gt_vel)
        local d_obj_prop = model.identitycriterion:backward(p_obj_prop,
                                                            gt_obj_prop)
        local d_pred = splitter:backward({prediction},
                                        {d_pos, d_vel, d_obj_prop})

        local g_input = model.network:backward(input, d_pred)
        if torch.find(mask,1)[1] == 1 then g_input = g_input[1] end
        preproc:updateGradInput(inp, g_input)

        collectgarbage()
        return loss, preproc.gradInput[1]  -- this should have been updated
    end

    local b2i_optstate = {learningRate = 0.01}

    -- infer the masses of ALL THE BALLS (or just you?)
    -- for now let's just infer the mass of you

    -- or should I preface the network with a wrapper that selects the input, because rmsprop expects a tensor!

    -- perturb
    -- this_past:resize(mp.batch_size, mp.num_past, mp.object_dim)
    -- this_past[{{},{},{5}}]:fill(1)
    -- this_past[{{},{},{6,8}}]:fill(0)
    -- this_past:resize(mp.batch_size, mp.num_past*mp.object_dim)

    print('initial input')
    print(this_past[{{1},{1,9}}])
    t = 1
    while t <= 100 do
        local old_this = this_past:clone()

        -- pass in input to rmsprop: automatically modifies this_past
        local _, loss = optim.rmsprop(feval_back2mass, this_past, b2i_optstate)  -- not getting updates! (or implicilty updated)

        -- modify only the mass
        old_this:resize(mp.batch_size, mp.num_past, mp.object_dim)
        this_past:resize(mp.batch_size, mp.num_past, mp.object_dim)

        -- [{{5,8}}] is the one-hot mass
        this_past[{{},{},{1,4}}] = old_this[{{},{},{1,4}}]
        this_past[{{},{},{9}}] = old_this[{{},{},{9}}]

        this_past:resize(mp.batch_size, mp.num_past*mp.object_dim)

        if t % 10 == 0 then
            b2i_optstate.learningRate = b2i_optstate.learningRate * 0.99
        end

        if t % 10 == 0 then
            print(this_past[{{1},{1,9}}])
        end

        if t % 10 == 0 then
            print(loss[1])
        end
        -- if (this_past-target_input):norm() < 1e-5 then
        --     break
        -- end
        t = t + 1
    end
    print ('final input after '..t..' iterations')
    print(this_past[{{1},{1,9}}])
end


function update_position(this, pred)
    -- this: (mp.batch_size, mp.num_past, mp.object_dim)
    -- prediction: (mp.batch_size, mp.num_future, mp.object_dim)
    -- pred is with respect to this[{{},{-1}}]
    ----------------------------------------------------------------------------
    local px = config_args.si.px
    local py = config_args.si.py
    local vx = config_args.si.vx
    local vy = config_args.si.vy
    local pnc = config_args.position_normalize_constant
    local vnc = config_args.velocity_normalize_constant

    local this, pred = this:clone(), pred:clone()
    local lastpos = (this[{{},{-1},{px,py}}]:clone()*pnc)
    local lastvel = (this[{{},{-1},{vx,vy}}]:clone()*vnc)  -- TODO: this is without subsampling make sure that this is correct!
    local currpos = (pred[{{},{},{px,py}}]:clone()*pnc)
    local currvel = (pred[{{},{},{vx,vy}}]:clone()*vnc)

    -- this is length n+1
    local pos = torch.cat({lastpos, currpos},2)
    local vel = torch.cat({lastvel, currvel},2)

    -- iteratively update pos through time. 
    for i = 1,pos:size(2)-1 do
        pos[{{},{i+1},{}}] = pos[{{},{i},{}}] + vel[{{},{i},{}}]  -- last dim=2
    end

    -- normalize again
    pos = pos/pnc
    assert(pos[{{},{1},{}}]:size(1) == pred:size(1))

    pred[{{},{},{px,py}}] = pos[{{},{2,-1},{}}]  -- reassign back to pred
    return pred
end

function update_angle(this, pred)
    local a = config_args.si.a
    local av = config_args.si.av
    local anc = config_args.angle_normalize_constant

    local this, pred = this:clone(), pred:clone()

    -- TODO: use config!
    local last_angle = this[{{},{-1},{a}}]:clone()*anc
    local last_angular_velocity = this[{{},{-1},{av}}]:clone()*anc
    local curr_angle = pred[{{},{},{a}}]:clone()*anc
    local curr_angular_velocity = pred[{{},{},{av}}]:clone()*anc

    -- this is length n+1
    local ang = torch.cat({last_angle, curr_angle},2)
    local ang_vel = torch.cat({last_angular_velocity, curr_angular_velocity},2)

    -- iteratively update ang through time. 
    for i = 1,ang:size(2)-1 do
        ang[{{},{i+1},{}}] = ang[{{},{i},{}}] + ang_vel[{{},{i},{}}]  -- last dim=2
    end

    -- normalize again
    ang = ang/config_args.angle_normalize_constant
    assert(ang[{{},{1},{}}]:size(1) == pred:size(1))

    pred[{{},{},{a}}] = ang[{{},{2,-1},{}}]  -- reassign back to pred
    return pred
end

function simulate_all(dataloader, params_, saveoutput, numsteps, gt)
    -- simulate two balls for now, but this should be made more general
    -- actually you should make this more general.
    -- there is no concept of ground truth here?
    -- or you can make the ground truth go as far as there are timesteps available
    --------------------------------------------------- -------------------------
    local avg_loss = 0
    local count = 0

    local unsqueezer = nn.Unsqueeze(2,3)
    if mp.cuda then unsqueezer:cuda() end

    assert(numsteps <= dataloader.maxwinsize-mp.num_past,
            'Number of predictive steps should be less than '..
            dataloader.maxwinsize-mp.num_past+1)
    for i = 1, dataloader.num_batches do
        if mp.server == 'pc' then xlua.progress(i, dataloader.num_batches) end

        local batch, current_dataset = dataloader:sample_sequential_batch(false)  -- TODO: perhaps here I should tell it what my desired windowsize should be

        -- get data
        local this_orig, context_orig, y_orig, context_future_orig, mask = unpack(batch)  -- NOTE CHANGE BATCH HERE

        -- crop to number of timestesp
        y_orig = y_orig[{{},{1, numsteps}}]
        context_future_orig = context_future_orig[{{},{},{1, numsteps}}]

        local num_particles = torch.find(mask,1)[1] + 1

        -- arbitrary notion of ordering here
        -- past: (bsize, num_particles, mp.numpast*mp.objdim)
        -- future: (bsize, num_particles, (mp.winsize-mp.numpast), mp.objdim)
        local past = torch.cat({unsqueezer:forward(this_orig:clone()), context_orig},2)
        local future = torch.cat({unsqueezer:forward(y_orig:clone()), context_future_orig},2)

        assert(past:size(2) == num_particles and future:size(2) == num_particles)

        local pred_sim = model_utils.transfer_data(
                            torch.zeros(mp.batch_size, num_particles,
                                        numsteps, mp.object_dim),
                            mp.cuda)

        -- loop through time
        for t = 1, numsteps do

            -- for each particle, update to the next timestep, given
            -- the past configuration of everybody
            -- total_particles = total_particles+num_particles

            for j = 1, num_particles do
                -- construct batch
                local this = torch.squeeze(past[{{},{j}}])

                local context
                if j == 1 then
                    context = past[{{},{j+1,-1}}]
                elseif j == num_particles then
                    context = past[{{},{1,-2}}]
                else
                    context = torch.cat({past[{{},{1,j-1}}],
                                                        past[{{},{j+1,-1}}]},2)
                end

                local y = future[{{},{j},{t}}]
                y:resize(mp.batch_size, mp.num_future, mp.object_dim)

                local batch = {this, context, y, _, mask} -- TODO: this may be the problem!

                -- predict
                local loss, pred = model:fp(params_,batch,true)   -- NOTE CHANGE THIS!
                avg_loss = avg_loss + loss
                count = count + 1

                pred = pred:reshape(mp.batch_size, mp.num_future, mp.object_dim)
                this = this:reshape(mp.batch_size, mp.num_past, mp.object_dim)  -- unnecessary

                -- -- relative coords for next timestep
                if mp.relative then
                    pred = data_process.relative_pair(this, pred, true)
                end

                -- restore object properties because we aren't learning them
                pred[{{},{},{config_args.ossi,-1}}] = this[{{},{-1},{config_args.ossi,-1}}]  -- NOTE! THIS DOESN'T TAKE ANGLE INTO ACCOUNT!
                
                -- update position
                pred = update_position(this, pred)

                -- update angle
                pred = update_angle(this, pred)
                pred = unsqueezer:forward(pred)

                -- write into pred_sim
                pred_sim[{{},{j},{t},{}}] = pred
            end

            -- update past for next timestep
            if mp.num_past > 1 then
                past = torch.cat({past[{{},{},{2,-1},{}}],
                                    pred_sim[{{},{},{t},{}}]}, 3)
            else
                assert(mp.num_past == 1)
                past = pred_sim[{{},{},{t},{}}]:clone()
            end

            -- local this_orig, context_orig, y_orig, context_future_orig, this_pred, context_future_pred, loss = model:sim(batch)

            
        end
        --- to be honest I don't think we need to break into past and context
        -- future, but actually that might be good for coloriing past and future, but
        -- actually I don't think so. For now let's just adapt it

        -- at this point, pred_sim should be all filled out
        -- break pred_sim into this and context_future
        -- recall: pred_sim: (batch_size,seq_length+1,numsteps,object_dim)
        -- recall that you had defined this_pred as the first obj in the future tensor
        local this_pred = torch.squeeze(pred_sim[{{},{1}}])
        if numsteps == 1 then this_pred = unsqueeze(this_pred,2) end

        local context_pred = pred_sim[{{},{2,-1}}]

        if mp.relative then
            y_orig = data_process.relative_pair(this_orig, y_orig, true)
        end

        -- local this_orig, context_orig, y_orig, context_future_orig, this_pred, context_future_pred, loss = model:sim(batch)

        if saveoutput and i <= mp.ns then
            save_ex_pred_json({this_orig, context_orig,
                                y_orig, context_future_orig,
                                this_pred, context_pred},
                                'batch'..test_loader.datasamplers[current_dataset].current_sampled_id..'.json',
                                current_dataset)
        end
    end
    avg_loss = avg_loss/count
    print('Mean Squared Error', avg_loss)
    collectgarbage()
end

-- eventually move this to variable_object_model
function inspect_hidden_state(dataloader, params_)
    local all_euc_dist = {}
    local all_effects_norm = {}
    for i = 1, dataloader.num_batches do
        local batch, current_dataset = dataloader:sample_sequential_batch(false)  -- actually you'd do this for multiple batches
        local euc_dist = get_euc_dist(batch[1], batch[2]) -- table of length num_context of {bsize}

        local loss, pred = model:fp(params_,batch,true)
        local effects = model.network:listModules()[2].output  -- effects[1] corresponds to context[{{},{1}}]  (bsize, rnn_dim)
        local effects_norm = {}  -- table of length num_context of {bsize}
        for j=1,#effects do
            table.insert(effects_norm, torch.squeeze(effects[j]:norm(2,2)))  -- you want to normalize in batch mode!
        end

        -- joining the two tables (but if you want to do individaul analysis you wouldn't do this)
        tablex.insertvalues(all_euc_dist,euc_dist)
        tablex.insertvalues(all_effects_norm,effects_norm)
    end
    all_euc_dist = torch.cat(all_euc_dist)
    all_effects_norm = torch.cat(all_effects_norm)

    local fname = 'hidden_state_all_testfolders'
    torch.save(mp.savedir..'/'..fname..'.t7', {euc_dist=all_euc_dist, effects_norm=all_effects_norm})
    if mp.server == 'pc' then
        -- plot scatter plot. TODO: later move this to an independent function
        gnuplot.pngfigure(mp.savedir..'/'..fname..'.png')
        gnuplot.xlabel('Euclidean Distance')
        gnuplot.ylabel('Hidden State Norm')
        gnuplot.title('Pairwise Hidden State as a Function of Distance from Focus Object')  -- TODO
        gnuplot.plot(all_euc_dist, all_effects_norm, '+')
        gnuplot.plotflush()
        print('Saved plot to '..mp.savedir..'/'..fname..'.png')
    end
end

function plot_tensor(tensor, info, subsamplerate)
    local toplot = subsample(tensor, subsamplerate)
    gnuplot.pngfigure(info[1])
    gnuplot.xlabel(info[2])
    gnuplot.ylabel(info[3])
    gnuplot.title(info[4])  -- change
    gnuplot.plot(unpack(toplot))
    gnuplot.plotflush()
end

-- return a table of euc dist between this and each of context
-- size is the number of items in context
-- is this for the last timestep of this?
-- TODO: later we can plot for all timesteps
function get_euc_dist(this, context, t)
    local num_context = context:size(2)
    local t = t or -1  -- efault use last timestep
    local this_pos = this[{{},{t},{1,2}}]  -- TODO: use config args
    local context_pos = context[{{},{},{t},{1,2}}]
    local diff = torch.squeeze(context_pos - this_pos:repeatTensor(1,num_context,1)) -- (bsize, num_context, 2)
    local diffsq = torch.pow(diff,2)
    local euc_dists = torch.sqrt(diffsq[{{},{},{1}}]+diffsq[{{},{},{2}}])  -- (bsize, num_context, 1)
    euc_dists = torch.split(euc_dists, 1,2)  --convert to table of (bsize, 1, 1)
    for i=1,#euc_dists do
        euc_dists[i] = torch.squeeze(euc_dists[i])
    end
    return euc_dists
end

function save_ex_pred_json(example, jsonfile, current_dataset)
    -- local flags = pls.split(mp.dataset_folder, '_')
    local flags = pls.split(mp.test_dataset_folders[current_dataset], '_')

    local world_config = {
        num_past = mp.num_past,
        num_future = mp.num_future,
        env=flags[1],--test_loader.scenario,
        numObj=tonumber(extract_flag(flags, 'n')),
        gravity=false, -- TODO
        friction=false, -- TODO
        pairwise=false -- TODO
    }

    -- first join on the time axis
    -- you should save context pred as well as context future
    local this_past, context_past,
            this_future, context_future,
            this_pred, context_pred = unpack(example)

    -- local subfolder = mp.savedir .. '/' .. 'predictions/'
    local subfolder = mp.savedir .. '/' .. mp.test_dataset_folders[current_dataset] .. 'predictions/'
    if not paths.dirp(subfolder) then paths.mkdir(subfolder) end

    -- construct gnd truth (could move to this to a util function)
    local this_pred_traj = torch.cat({this_past, this_pred}, 2)
    local context_pred_traj = torch.cat({context_past,context_pred}, 3)
    dp:record_trajectories({this_pred_traj, context_pred_traj}, world_config, subfolder..'pred_' .. jsonfile)

    -- construct prediction
    local this_gt_traj = torch.cat({this_past, this_future}, 2)
    local context_gt_traj = torch.cat({context_past, context_future}, 3)
    dp:record_trajectories({this_gt_traj, context_gt_traj}, world_config, subfolder..'gt_' .. jsonfile)
end

function getLastSnapshot(network_name)

    -- TODO: this should be replaced by savedir!
    local res_file = io.popen("ls -t "..mp.logs_root..'/'..network_name..
                                " | grep -i epoch | head -n 1")
    local status, result = pcall(function()
                return res_file:read():match( "^%s*(.-)%s*$" ) end)
    print(result)
    res_file:close()
    if not status then
        return false
    else
        return result
    end
end

function run_inspect_hidden_state()
    print(mp.name)
    print(string.gsub(mp.name,"'","\'"))

    local snapshot = getLastSnapshot(mp.name)
    local snapshotfile = mp.savedir ..'/'..snapshot
    print(snapshotfile)
    local checkpoint = torch.load(snapshotfile)

    local saved_args = torch.load(mp.savedir..'/args.t7')
    mp = merge_tables(saved_args.mp, mp) -- overwrite saved mp with our mp when applicable
    config_args = saved_args.config_args

    model_deps(mp.model)
    inittest(true, snapshotfile, {sim=false})  -- assuming the mp.savedir doesn't change

    print(inspect_hidden_state(test_loader, checkpoint.model.theta.params, true, mp.steps))
end


function predict_simulate_all()
    print(mp.name)
    print(string.gsub(mp.name,"'","\'"))

    local snapshot = getLastSnapshot(mp.name)
    local snapshotfile = mp.savedir ..'/'..snapshot
    print(snapshotfile)
    local checkpoint = torch.load(snapshotfile)

    local saved_args = torch.load(mp.savedir..'/args.t7')
    mp = merge_tables(saved_args.mp, mp) -- overwrite saved mp with our mp when applicable
    config_args = saved_args.config_args

    model_deps(mp.model)
    inittest(true, snapshotfile, {sim=true})  -- assuming the mp.savedir doesn't change
    print(simulate_all(test_loader, checkpoint.model.theta.params, true, mp.steps))
end

function predict_b2i()
    local snapshot = getLastSnapshot(mp.name)
    local snapshotfile = mp.savedir ..'/'..snapshot
    local checkpoint = torch.load(snapshotfile)
    checkpoint = checkpointtocuda(checkpoint)
    mp = merge_tables(checkpoint.mp, mp)
    model_deps(mp.model)
    inittest(true, snapshotfile)  -- assuming the mp.savedir doesn't change
    backprop2input()
end

function model_deps(modeltype)
    if modeltype == 'lstmobj' or
            modeltype == 'ffobj' or
                    modeltype == 'gruobj' then
        M = require 'variable_obj_model'
    elseif modeltype == 'lstmtime' then
        M = require 'lstm_model'
    elseif modeltype == 'ff' then
        M = require 'feed_forward_model'
    else
        error('Unrecognized model')
    end
end


------------------------------------- Main -------------------------------------
if mp.mode == 'sim' then
    predict_simulate_all()
elseif mp.mode == 'hid' then
    run_inspect_hidden_state()
elseif mp.mode == 'b2i' then
    predict_b2i()
elseif mp.mode == 'pred' then
    predict()
else
    error('unknown mode')
end
