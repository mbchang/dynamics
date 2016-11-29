-- Michael B Chang

-- Third Party Imports
require 'torch'
require 'nn'
require 'optim'
require 'image'
require 'xlua'
require 'sys'
require 'pl'
require 'torchx'
torch.setdefaulttensortype('torch.FloatTensor')
require 'data_utils'
local tablex = require 'pl.tablex'
local pls = require 'pl.stringx'

require 'rnn'
require 'torch'
require 'nngraph'
require 'Base'
require 'IdentityCriterion'
require 'gnuplot'

-- Local Imports
local model_utils = require 'model_utils'
local D = require 'general_data_sampler'
local D2 = require 'datasaver'
require 'logging_utils'
require 'json_interface'

local data_process = require 'data_process'

------------------------------------- Init -------------------------------------
local cmd = torch.CmdLine()
cmd:option('-mode', "exp", 'sim | tva')
cmd:option('-server', "op", 'pc = personal | op = openmind')
cmd:option('logs_root', 'logs', 'subdirectory to save logs and checkpoints')
cmd:option('data_root', '../../data', 'subdirectory to save data')
cmd:option('-name', "", 'experiment name')
cmd:option('-seed', true, 'manual seed or not')

-- dataset
cmd:option('-test_dataset_folders', '', 'dataset folder')

-- experiment options
cmd:option('-ns', 3, 'number of test batches')
cmd:option('-steps', 58, 'steps to simulate')
cmd:text()

-- parse input params
mp = cmd:parse(arg)

if mp.server == 'pc' then
    -- mp.data_root = '../.'    
    mp.logs_root = 'logs'
    mp.winsize = 3 -- total number of frames
    mp.num_past = 2 --10
    mp.num_future = 1 --10
	mp.batch_size = 5 --1
	mp.num_threads = 1
	mp.cuda = false
else
	mp.winsize = 3  -- total number of frames
    mp.num_past = 2 -- total number of past frames
    mp.num_future = 1
	mp.num_threads = 4
	mp.cuda = false
end

local M

-- world constants
local subsamp = 1


mp.name = string.gsub(string.gsub(string.gsub(mp.name,'{',''),'}',''),"'",'')
mp.test_dataset_folders = assert(loadstring("return "..string.gsub(mp.test_dataset_folders,'\"',''))())
mp.savedir = mp.logs_root .. '/' .. mp.name
mp.relative=true

if mp.seed then torch.manualSeed(123) end
if mp.cuda then
    require 'cutorch'
    require 'cunn'
end

local model, test_loader, modelfile, dp

------------------------------- Helper Functions -------------------------------

function inittest(preload, model_path, opt)
    dp = data_process.create(model_path, model_path, config_args)
    model = M.create(mp, preload, model_path)
    mp.cuda = false

    if not(string.find(mp.savedir, 'tower') == nil) then
        assert((string.find(mp.savedir, 'ball') == nil) and 
               (string.find(mp.savedir, 'mixed') == nil) and 
               (string.find(mp.savedir, 'invisible') == nil))
        config_args.maxwinsize = config_args.maxwinsize_long
    else
        config_args.maxwinsize = config_args.maxwinsize
    end

    local data_loader_args = {data_root=mp.data_root..'/',
                              dataset_folders=mp.test_dataset_folders,
                              maxwinsize=config_args.maxwinsize,
                              winsize=mp.winsize,
                              num_past=mp.num_past,
                              num_future=mp.num_future,
                              relative=mp.relative,
                              subdivide=opt.subdivide,
                              shuffle=config_args.shuffle,
                              sim=opt.sim,
                              cuda=mp.cuda
                            }
    test_loader = D.create('testset', tablex.deepcopy(data_loader_args))

    modelfile = model_path
    print("Network parameters:")
    print(mp)
    print(model.network)
end


local function simulate_all_preprocess(past, future, j, t, num_particles)
    -- construct batch
    local this = torch.squeeze(past[{{},{j}}])

    local y = future[{{},{j},{t}}]
    y = torch.squeeze(y,2)

    local y_before_relative = y:clone()

    if mp.relative then
        y = data_process.relative_pair(this, y, false)  -- absolute to relative
    end

    local context, context_future
    if j == 1 then
        context = past[{{},{j+1,-1}}]
        context_future = future[{{},{j+1,-1},{t}}]
    elseif j == num_particles then
        context = past[{{},{1,-2}}]
        context_future = future[{{},{1,-2},{t}}]
    else
        context = torch.cat({past[{{},{1,j-1}}], past[{{},{j+1,-1}}]},2)
        context_future = torch.cat({future[{{},{1,j-1},{t}}], future[{{},{j+1,-1},{t}}]},2)
    end

    -- return this, y, context, context_future, y_before_relative
    local max_obj = 12
    local trimmed_context, closest_indices = data_process.k_nearest_context(this:clone(), context:clone(), max_obj)
    local trimmed_context_future = data_process.k_nearest_context(y_before_relative:clone(), context_future:clone(), max_obj)

    return this, y, trimmed_context, trimmed_context_future, y_before_relative
end

local function simulate_all_postprocess(pred, this, raw_obj_dim)
    -- HERE chop off the last part in pred
    if mp.of then pred = pred[{{},{1,-2}}] end

    pred = pred:reshape(mp.batch_size, mp.num_future, raw_obj_dim)

    -- relative coords for next timestep
    if mp.relative then
        pred = data_process.relative_pair(this, pred, true)  -- relative to absolute
    end

    -- restore object properties because we aren't learning them
    pred[{{},{},{config_args.ossi,-1}}] = this[{{},{-1},{config_args.ossi,-1}}]

    -- update position
    pred = model:update_position(this, pred)

    -- update angle
    pred = model:update_angle(this, pred)

    -- if object is ball, then angle and angular velocity are 0
    if pred[{{},{},config_args.si.oid[1]}]:equal(convert_type(torch.ones(mp.batch_size,1), mp.cuda)) then
        pred[{{},{},{config_args.si.a,config_args.si.av}}]:zero()
    end

    pred = unsqueeze(pred, 2)
    return pred
end

-- invalid_focus_mask: 1 means invalid, 0 valid
local function make_invalid_dummy(this, invalid_focus_mask)
    local this = this:clone()
    assert(invalid_focus_mask:sum() > 0)

    -- first we find a valid focus object. that will be our dummy.
    -- we find a zero element
    local dummy_idx = torch.find(invalid_focus_mask,0)[1]
    local dummy_focus = this[{{dummy_idx},{},{}}]:clone() 

    -- then, for all invalid focus object, we will replace it with the dummy
    local invalid_idxs = torch.find(invalid_focus_mask,1)
    for _,invalid_idx in pairs(invalid_idxs) do
        this[{{invalid_idx},{},{}}] = dummy_focus:clone()
    end

    return this
end 


local function replace_invalid_dummy(pred, y_before_relative, this, invalid_focus_mask)
    local pred = pred:clone()
    local y_before_relative = y_before_relative:clone()
    local this = this:clone()
    assert(invalid_focus_mask:sum() > 0)

    -- next we will take 
    local invalid_idxs = torch.find(invalid_focus_mask,1)
    for _,invalid_idx in pairs(invalid_idxs) do
        pred[{{invalid_idx},{},{}}] = y_before_relative[{{invalid_idx},{},{}}]:clone() -- replace with ground truth
    end

    -- wait, note that you still need ground truth in order to compute your prediction score. so it is okay
    -- to predict against a ground truth here.
    return pred  
end


local function apply_mask_avg(tensor, mask)
    local mask = torch.squeeze(mask:clone())
    local masked = torch.cmul(tensor:clone(), mask:float())
    local num_valid = mask:sum()
    local averaged = masked:sum()/num_valid
    return averaged, num_valid
end

local function find_valid_focus_mask(this)
    -- Ok, note that you only want the examples where this is a ball or block
    -- these templates are (bsize, oid_dim)
    local oid_onehot, template_ball, template_block, template_obstacle = get_oid_templates(this, config_args, mp.cuda)
    local num_oids = config_args.si.oid[2]-config_args.si.oid[1]+1
    local invalid_focus_mask = oid_onehot:eq(template_obstacle):sum(2):eq(num_oids)  -- 1 if invalid
    local valid_focus_mask = 1-invalid_focus_mask -- 1 if valid
    if invalid_focus_mask:sum() > 0 then has_invalid_focus = true end -- NOTE added this!
    return valid_focus_mask, invalid_focus_mask
end


function simulate_all(dataloader, params_, saveoutput, numsteps, gt)
    local losses_through_time_all_batches = torch.zeros(dataloader.total_batches, numsteps)
    local mag_error_through_time_all_batches = torch.zeros(dataloader.total_batches, numsteps)
    local ang_error_through_time_all_batches = torch.zeros(dataloader.total_batches, numsteps)
    local vel_loss_through_time_all_batches = torch.zeros(dataloader.total_batches, numsteps)
    local ang_vel_loss_through_time_all_batches = torch.zeros(dataloader.total_batches, numsteps)

    local experiment_name = paths.basename(dataloader.dataset_folder)
    local subfolder = mp.savedir .. '/' .. experiment_name .. '_predictions/'
    if not paths.dirp(subfolder) then paths.mkdir(subfolder) end

    local logfile = 'gt_divergence.log'
    local gtdivergenceLogger = optim.Logger(paths.concat(subfolder, logfile))  -- this should be dataloader specific!
    gtdivergenceLogger.showPlot = false
    -- I have to average through all batches

    assert(numsteps <= dataloader.maxwinsize-mp.num_past,
            'Number of predictive steps should be less than '..
            dataloader.maxwinsize-mp.num_past+1)
    for i = 1, dataloader.total_batches do

        if mp.server == 'pc' then xlua.progress(i, dataloader.total_batches) end

        local batch = dataloader:sample_sequential_batch()

        -- get data
        local this_orig, context_orig, y_orig, context_future_orig, mask, original_batch, trimmed_context_indices = unpack(batch)  -- no flag yet

        -- in original batch we have
        local untrimmed_context_past = original_batch[2]
        local untrimmed_context_future = original_batch[4]
        local untrimmed_context_future_orig = untrimmed_context_future:clone()
        local has_invalid_focus = false
        -- context_orig and context_future_orig correspond 

        local raw_obj_dim = this_orig:size(3)

        -- crop to number of timestesp
        y_orig = y_orig[{{},{1, numsteps}}]   -- no flag yet
        context_future_orig = context_future_orig[{{},{},{1, numsteps}}]   -- no flag yet

        local num_particles = torch.find(mask,1)[1] + 1

        -- arbitrary notion of ordering here
        -- past: (bsize, num_particles, mp.numpast*mp.objdim)
        -- future: (bsize, num_particles, (mp.winsize-mp.numpast), mp.objdim)
        -- local past = torch.cat({unsqueeze(this_orig:clone(),2), context_orig},2)   -- no flag yet
        -- local future = torch.cat({unsqueeze(y_orig:clone(),2), context_future_orig},2)     -- no flag yet (because we don't know which is focus or context)

        local past = torch.cat({unsqueeze(this_orig:clone(),2), untrimmed_context_past:clone()},2)   -- no flag yet
        local future = torch.cat({unsqueeze(y_orig:clone(),2), untrimmed_context_future:clone()},2)     -- no flag yet (because we don't know which is focus or context)

        local num_particles = past:size(2)

        local pred_sim = model_utils.transfer_data(
                            torch.zeros(mp.batch_size, num_particles,
                                        numsteps, y_orig:size(3)),   -- the dimensionality here inludes a flag; this may be a problem?
                            mp.cuda)

        -- loop through time
        for t = 1, numsteps do
            if mp.server == 'pc' then xlua.progress(t, numsteps) end

            -- for each particle, update to the next timestep, given
            -- the past configuration of everybody

            -- it makes no sense to accumulate
            local loss_within_batch = 0
            local mag_error_within_batch = 0
            local ang_error_within_batch = 0
            local vel_loss_within_batch = 0
            local ang_vel_loss_within_batch = 0
            local counter_within_batch = 0
            local angmag_counter_within_batch = 0

            for j = 1, num_particles do

                -- switch to a new focus object
                local this, y, context, context_future, y_before_relative = simulate_all_preprocess(past, future, j, t, num_particles)

                -- Ok, note that you only want the examples where this is a ball or block
                -- these templates are (bsize, oid_dim)
                local valid_focus_mask, invalid_focus_mask = find_valid_focus_mask(this)

                if invalid_focus_mask:sum() > 0 then has_invalid_focus = true end -- NOTE added this!

                if invalid_focus_mask:sum() < invalid_focus_mask:size(1) then
                    -- note that we have to keep the batch size constant.
                    -- okay, so I think we'd need to do some dummy filling.
                    -- for the ones that have an obstacle, I just need to fill it with a dummy entry
                    -- then after I predict, I replace it with its corresponding entry in future, but remember to apply relative pair. 
                    -- to check, make sure that the context obstacles just never move.

                    -- if we have some entries where obstacle is this
                    if invalid_focus_mask:sum() > 0 then
                        -- for the ones that have an obstacle, I just need to fill it with a dummy entry
                        -- note that we didn't change the context (we should for a valid prediction) but we will replace the prediction anyways
                        this = make_invalid_dummy(this, invalid_focus_mask:clone())
                    end 

                    -- construct batch
                    local batch = {this, context, y, _, mask}  -- you need context_future to be in here!

                    local loss_batch, pred, vel_loss_batch, ang_vel_loss_batch = model:fp_batch(params_,batch,true)

                    local loss = apply_mask_avg(loss_batch, valid_focus_mask)
                    local vel_loss = apply_mask_avg(vel_loss_batch, valid_focus_mask)
                    local ang_vel_loss = apply_mask_avg(ang_vel_loss_batch, valid_focus_mask)

                    local angle_error_batch, relative_magnitude_error_batch, angle_mask = angle_magnitude(pred, batch, true)

                    -- note that angle_mask is applied over batch_size. 
                    local valid_focus_angle_mask = torch.cmul(valid_focus_mask,angle_mask) -- correct
                    local angle_error = apply_mask_avg(angle_error_batch, valid_focus_angle_mask)
                    local relative_magnitude_error = apply_mask_avg(relative_magnitude_error_batch, valid_focus_angle_mask)

                    -- record
                    loss_within_batch = loss_within_batch + loss
                    vel_loss_within_batch = vel_loss_within_batch + vel_loss
                    ang_vel_loss_within_batch = ang_vel_loss_within_batch + ang_vel_loss
                    ang_error_within_batch = ang_error_within_batch + angle_error
                    mag_error_within_batch = mag_error_within_batch + relative_magnitude_error

                    counter_within_batch = counter_within_batch + valid_focus_mask:sum()/valid_focus_mask:size(1) -- actually this should be the fraction of valid examples
                    angmag_counter_within_batch = angmag_counter_within_batch + valid_focus_angle_mask:sum()/valid_focus_angle_mask:size(1)

                    -- update non-predictive parts of pred
                    pred = simulate_all_postprocess(pred, this, raw_obj_dim)

                    -- here you should apply the mask (such that by the end of it pred_sim will look valid)
                    if invalid_focus_mask:sum() > 0 then

                        -- relative y
                        pred = replace_invalid_dummy(pred, y_before_relative, this, invalid_focus_mask:clone()) 
                    end

                    -- write into pred_sim
                    pred_sim[{{},{j},{t},{}}] = pred

                else
                    -- we only reach here IF all of the FOCUS objects are INVALID
                    assert((torch.squeeze(this[{{},{-1}}])-y_before_relative):norm()==0)  -- they had better be the same if they are stationary (we assume they can't move)
                    pred_sim[{{},{j},{t},{}}] = y_before_relative -- but does this contain the context though?
                end
            end

            -- update past for next timestep
            if mp.num_past > 1 then
                past = torch.cat({past[{{},{},{2,-1},{}}], pred_sim[{{},{},{t},{}}]}, 3)
            else
                assert(mp.num_past == 1)
                past = pred_sim[{{},{},{t},{}}]:clone()
            end

            -- record
            losses_through_time_all_batches[{{i},{t}}] = loss_within_batch/counter_within_batch
            vel_loss_through_time_all_batches[{{i},{t}}] = vel_loss_within_batch/counter_within_batch
            ang_vel_loss_through_time_all_batches[{{i},{t}}] = ang_vel_loss_within_batch/counter_within_batch

            ang_error_through_time_all_batches[{{i},{t}}] = ang_error_within_batch/angmag_counter_within_batch
            mag_error_through_time_all_batches[{{i},{t}}] = mag_error_within_batch/angmag_counter_within_batch
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

        if saveoutput and i <= mp.ns then
            save_ex_pred_json({this_orig, untrimmed_context_past,
                                y_orig, untrimmed_context_future_orig,
                                this_pred, context_pred},
                                'batch'..dataloader.current_sampled_id..'.json',
                                experiment_name,
                                subfolder)
        end

        collectgarbage()
    end

    -- average over all the batches
    local averaged_losses_through_time_all_batches = torch.totable(torch.squeeze(losses_through_time_all_batches:mean(1)))
    local averaged_ang_error_through_time_all_batches = torch.totable(torch.squeeze(ang_error_through_time_all_batches:mean(1)))
    local averaged_mag_error_through_time_all_batches = torch.totable(torch.squeeze(mag_error_through_time_all_batches:mean(1)))
    local averaged_vel_loss_through_time_all_batches = torch.totable(torch.squeeze(vel_loss_through_time_all_batches:mean(1)))
    local averaged_ang_vel_loss_through_time_all_batches = torch.totable(torch.squeeze(ang_vel_loss_through_time_all_batches:mean(1)))

    print('averaged_losses_through_time_all_batches')
    print(averaged_losses_through_time_all_batches)

    print('averaged_ang_error_through_time_all_batches')
    print(averaged_ang_error_through_time_all_batches)

    print('averaged_mag_error_through_time_all_batches')
    print(averaged_mag_error_through_time_all_batches)

    print('averaged_vel_loss_through_time_all_batches')
    print(averaged_vel_loss_through_time_all_batches)

    print('averaged_ang_vel_loss_through_time_all_batches')
    print(averaged_ang_vel_loss_through_time_all_batches)

    for tt=1,#averaged_losses_through_time_all_batches do
        gtdivergenceLogger:add{['Timesteps'] = tt, 
                                ['MSE Error'] = averaged_losses_through_time_all_batches[tt],
                                ['Cosine Difference'] = averaged_ang_error_through_time_all_batches[tt],
                                ['Magnitude Difference'] = averaged_mag_error_through_time_all_batches[tt],
                                ['Velocity Error'] = averaged_vel_loss_through_time_all_batches[tt],
                                ['Angular Velocity Error'] = averaged_ang_vel_loss_through_time_all_batches[tt]
                            }
        gtdivergenceLogger:style{['Timesteps'] = '~',
                                ['MSE Error'] = '~',
                                ['Cosine Difference'] = '~',
                                ['Magnitude Difference'] = '~',
                                ['Velocity Error'] = '~',
                                ['Angular Velocity Error'] = '~',
                            }
    end
    collectgarbage()
end


-- TODO_lowpriority: move this to plot_results
function plot_hid_state(fname, x,y)
    gnuplot.pngfigure(mp.savedir..'/'..fname..'.png')
    gnuplot.xlabel('Euclidean Distance')
    gnuplot.ylabel('Hidden State Norm')
    gnuplot.title('Pairwise Hidden State as a Function of Distance from Focus Object')
    gnuplot.plot(x, y, '+')
    gnuplot.plotflush()
    print('Saved plot of hidden state to '..mp.savedir..'/'..fname..'.png')
end


function save_ex_pred_json(example, jsonfile, current_dataset, subfolder)
    print(current_dataset)
    local flags = pls.split(current_dataset, '_')

    local world_config = {
        num_past = mp.num_past,
        num_future = mp.num_future,
        env=flags[1],
        numObj=tonumber(extract_flag(flags, 'n')),
        gravity=false,
        friction=false,
        pairwise=false
    }

    -- first join on the time axis
    -- you should save context pred as well as context future
    local this_past, context_past,
            this_future, context_future,
            this_pred, context_pred = unpack(example)

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
    local res_file = io.popen("ls -t "..mp.logs_root..'/'..network_name..
                                " | grep -i epoch | head -n 1")
    local status, result = pcall(function()
                return res_file:read():match( "^%s*(.-)%s*$" ) end)
    print('Last Snapshot: '..result)
    res_file:close()
    if not status then
        return false
    else
        return result
    end
end

function predict_simulate_all()
    local checkpoint, snapshotfile = load_most_recent_checkpoint()

    inittest(true, snapshotfile, {sim=true, subdivide=false})  -- assuming the mp.savedir doesn't change
    require 'infer'
    print('Network parameters')
    print(mp)

    for i,testdataset_loader in pairs(test_loader.datasamplers) do
        print('Evaluating '..test_loader.dataset_folders[i])
        simulate_all(testdataset_loader, checkpoint.model.theta.params, true, mp.steps)
    end
end

function load_most_recent_checkpoint()
    local snapshot = getLastSnapshot(mp.name)
    return load_checkpoint(snapshot)
end

-- 'step9' for mixed
function load_specified_checkpoint(tag)
    local checkpoints = get_all_checkpoints(mp.logs_root, mp.name)
    for _,c in pairs(checkpoints) do
        if not(string.find(c, tag) == nil) then
            return load_checkpoint(c)
        end
    end
    assert(false, 'You should not reah this point')
end

function get_all_checkpoints(logs_folder, experiment_name)
    -- iterate through snapshots
    local res_file = io.popen("ls -t "..mp.logs_root..'/'..mp.name..
                            " | grep -i epoch")
    local checkpoints = {}
    while true do
        local result = res_file:read()
        if result == nil then break end
        print('Adding snapshot: '..result)
        table.insert(checkpoints, result)
    end

    return checkpoints
end

function load_checkpoint(snapshot)
    local snapshotfile = mp.savedir ..'/'..snapshot
    local checkpoint = torch.load(snapshotfile)
    local saved_args = torch.load(mp.savedir..'/args.t7')
    mp = merge_tables(saved_args.mp, mp) -- overwrite saved mp with our mp when applicable
    config_args = saved_args.config_args

    -- NOTE THIS IS ONLY FOR THE EXPERIMENTS THAT DON'T HAVE object_base_size_ids_upper!
    config_args.object_base_size_ids_upper={60,80*math.sqrt(2)/2,math.sqrt(math.pow(60,2)+math.pow(60/3,2))}

    model_deps(mp.model)
    return checkpoint, snapshotfile
end

function inference(logfile, property, method, cf)
    local checkpoints = get_all_checkpoints(mp.logs_root, mp.name)

    local inferenceLogger = optim.Logger(paths.concat(mp.savedir ..'/', logfile))
    inferenceLogger.showPlot = false

    -- iterate through checkpoints backwards (least recent to most recent)
    for i=#checkpoints,1,-1 do
        print(property..' inference on snapshot '..checkpoints[i])

        local checkpoint, snapshotfile = load_checkpoint(checkpoints[i])

        inittest(true, snapshotfile, {sim=false, subdivide=true})  -- assuming the mp.savedir doesn't change
        require 'infer'

        -- save num_correct into a file
        local accuracy, accuracy_by_speed, accuracy_by_mass = infer_properties(model, test_loader, checkpoint.model.theta.params, property, method, cf)
        print('Accuracy',accuracy)
        inferenceLogger:add{[property..' accuracy (test set)'] = accuracy}
        inferenceLogger:style{[property..' accuracy (test set)'] = '~'}
    end
    print('Finished '..property..' inference')
end

function property_analysis_all(logfile, property)
    local checkpoints = get_all_checkpoints(mp.logs_root, mp.name)

    local analysisLogger = optim.Logger(paths.concat(mp.savedir ..'/', logfile))
    analysisLogger.showPlot = false

    -- iterate through checkpoints backwards (least recent to most recent)
    for i=#checkpoints,1,-1 do
        print(' property analysis on snapshot '..checkpoints[i])

        local checkpoint, snapshotfile = load_checkpoint(checkpoints[i])

        inittest(true, snapshotfile, {sim=false, subdivide=true})  -- assuming the mp.savedir doesn't change
        require 'infer'

        local avg_property, num_property = property_analysis(model, test_loader, checkpoint.model.theta.params, property)
        print(avg_property, num_property)

        local metrics = {'loss', 'vel_loss', 'ang_loss', 'avg_ang_error', 'avg_rel_mag_error'}

        print('avg_property')
        local logger_table = {}
        local logger_table_style = {}
        for k,v in pairs(avg_property) do
            print(k)
            print(v)

            for m,n in pairs(torch.totable(torch.squeeze(v))) do
                print(metrics[m]..'_'..k)
                print(n)

                logger_table[metrics[m]..'_'..k] = n
                logger_table_style[metrics[m]..'_'..k] = '~'   
            end
        end

        print(logger_table)

        analysisLogger:add(logger_table)
        analysisLogger:style(logger_table_style)   
    end
    print('Finished property analysis')
end

local function test_vel_angvel(dataloader, params_, saveoutput, num_batches)
    local sum_loss = 0
    local num_batches = num_batches or dataloader.total_batches

    num_batches = math.min(5000, num_batches)
    print('Testing '..num_batches..' batches')
    local total_avg_vel = 0
    local total_avg_ang_vel = 0
    local total_avg_loss = 0
    local total_avg_ang_error = 0
    local total_avg_rel_mag_error = 0

    for i = 1,num_batches do
        if mp.server == 'pc' then xlua.progress(i, num_batches) end
        local batch = dataloader:sample_sequential_batch(false)

        -- may need to change this
        local loss, pred, avg_batch_vel, avg_batch_ang_vel  = model:fp(params_, batch)
        local avg_angle_error, avg_relative_magnitude_error = angle_magnitude(pred, batch)

        total_avg_vel = total_avg_vel+ avg_batch_vel
        total_avg_ang_vel = total_avg_ang_vel + avg_batch_ang_vel
        total_avg_loss = total_avg_loss + loss
        total_avg_ang_error = total_avg_ang_error + avg_angle_error
        total_avg_rel_mag_error = total_avg_rel_mag_error + avg_relative_magnitude_error
    end
    total_avg_vel = total_avg_vel/num_batches
    total_avg_ang_vel = total_avg_ang_vel/num_batches
    total_avg_loss = total_avg_loss/num_batches
    total_avg_ang_error = total_avg_ang_error/num_batches
    total_avg_rel_mag_error = total_avg_rel_mag_error/num_batches

    if mp.cuda then cutorch.synchronize() end
    collectgarbage()
    return total_avg_loss, total_avg_vel, total_avg_ang_vel, total_avg_ang_error, total_avg_rel_mag_error
end


function test_vel_angvel_all()
    local checkpoints = get_all_checkpoints(mp.logs_root, mp.name)

    local eval_data = {}

    -- iterate through checkpoints backwards (least recent to most recent)
    for i=#checkpoints,1,-1 do
        local checkpoint, snapshotfile = load_checkpoint(checkpoints[i])
        inittest(true, snapshotfile, {sim=false, subdivide=true})  -- assuming the mp.savedir doesn't change
        require 'infer'

        local checkpoint_eval_data = torch.zeros(#test_loader.datasamplers, 5)  -- (num_samplers, [avg_vel_loss, avg_ang_vel_loss])

        for i,testdataset_loader in pairs(test_loader.datasamplers) do
            print('Evaluating '..test_loader.dataset_folders[i])
            local avg_loss, avg_vel_loss, avg_ang_vel_loss, avg_ang_error, avg_rel_mag_error = test_vel_angvel(testdataset_loader, checkpoint.model.theta.params, false)
            print(avg_loss, avg_vel_loss, avg_ang_vel_loss, avg_ang_error, avg_rel_mag_error)
            checkpoint_eval_data[{{i},{}}] = torch.Tensor{avg_loss, avg_vel_loss, avg_ang_vel_loss, avg_ang_error, avg_rel_mag_error}
        end

        table.insert(eval_data, checkpoint_eval_data:clone():reshape(1,#test_loader.datasamplers,5))
    end

    -- need to transpose
    eval_data = torch.cat(eval_data,1)  -- (num_checkpoints, num_samplers, 5)
    eval_data = eval_data:transpose(1,2) -- (num_samplers, num_checkpoints, 5)

    print(eval_data)
    print(eval_data:gt(0))

    -- iterate through samplers
    for s,testdataset_loader in pairs(test_loader.datasamplers) do
        local experiment_name = paths.basename(testdataset_loader.dataset_folder)
        local subfolder = mp.savedir .. '/' .. experiment_name .. '_predictions/'
        if not paths.dirp(subfolder) then paths.mkdir(subfolder) end

        local logfile = 'tva.log'
        local tvaLogger = optim.Logger(paths.concat(subfolder, logfile))
        tvaLogger.showPlot = false

        for c=1,#checkpoints do
            tvaLogger:add{['loss'] = eval_data[{{s},{c},{1}}]:sum(), 
                          ['vel_loss'] = eval_data[{{s},{c},{2}}]:sum(),
                          ['ang_vel_loss'] = eval_data[{{s},{c},{3}}]:sum(),
                          ['avg_ang_error'] = eval_data[{{s},{c},{4}}]:sum(),
                          ['avg_rel_mag_error'] = eval_data[{{s},{c},{5}}]:sum()}
            tvaLogger:style{['loss'] = '~',
                            ['vel_loss'] = '~',
                            ['ang_vel_loss'] = '~',
                            ['avg_ang_error'] = '~',
                            ['avg_rel_mag_error'] = '~'}
        end
    end
end



function predict_test_first_timestep_all()
    local checkpoint, snapshotfile = load_most_recent_checkpoint()
    inittest(true, snapshotfile, {sim=false, subdivide=false})  -- assuming the mp.savedir doesn't change
    print('Network parameters')
    print(mp)

    for i,testdataset_loader in pairs(test_loader.datasamplers) do
        print('Evaluating '..test_loader.dataset_folders[i])
        local avg_loss, avg_vel_loss, avg_ang_vel_loss = test_vel_angvel(testdataset_loader, checkpoint.model.theta.params, true)
        print('avg_loss', avg_loss, 'avg_vel_loss', avg_vel_loss, 'avg_ang_vel_loss', avg_ang_vel_loss)
    end
end


function mass_inference()
    inference('mass_infer_cf.log', 'mass', 'max_likelihood', true)
end

function size_inference()
    inference('size_infer_cf.log', 'size', 'max_likelihood_context', true)
end

function objtype_inference()
    inference('objtype_infer_cf.log', 'objtype', 'max_likelihood_context', true)
end


function size_analysis()
    property_analysis_all('size_analysis.log', 'size')
end

function oid_analysis()
    property_analysis_all('oid_analysis.log', 'objtype')
end


function model_deps(modeltype)
    if modeltype == 'npe' then
        M = require 'npe'
    elseif modeltype == 'np' then
        M = require 'nop'
    elseif modeltype == 'lstm' then
        M = require 'lstm'
    else
        error('Unrecognized model')
    end
end


------------------------------------- Main -------------------------------------
if mp.mode == 'sim' then
    predict_simulate_all()
elseif mp.mode == 'minf' then
    mass_inference()
elseif mp.mode == 'sinf' then
    size_inference()
elseif mp.mode == 'oinf' then
    objtype_inference()
elseif mp.mode == 'pmofminf' then
    pmofm_b2i_inference()
elseif mp.mode == 'pred' then
    predict()
elseif mp.mode == 'tva' then
    test_vel_angvel_all()
elseif mp.mode == 'tf' then
    predict_test_first_timestep_all()
elseif mp.mode == 'pa' then
    property_analysis_all()
elseif mp.mode == 'sa' then
    size_analysis()
elseif mp.mode == 'oia' then
    oid_analysis()
else
    error('unknown mode')
end
