require 'torchx'
require 'json_interface'
require 'data_utils'
require 'paths'
require 'json'
require 'nn'
local pls = require 'pl.stringx'
local plt = require 'pl.tablex'

local data_process = {}
data_process.__index = data_process


function data_process.create(jsonfolder, outfolder, args) -- I'm not sure if this make sense in eval though
    local self = {}
    setmetatable(self, data_process)
    self.pnc = args.position_normalize_constant
    self.vnc = args.velocity_normalize_constant
    self.anc = args.angle_normalize_constant
    self.relative = args.relative -- bool
    self.masses = args.masses -- {0.33, 1.0, 3.0, 1e30}
    self.rsi = args.rsi -- {px: 1, py: 2, vx: 3, vy: 4, m: 5, oid: 6}
    self.si = args.si -- {px: {1}, py: {2}, vx: {3}, vy: {4}, m: {5,8}, oid: {9}}
    self.oid_ids = args.oid_ids
    self.boolean = args.boolean
    self.permute_context = args.permute_context  -- bool: if True will expand the dataset, False won't NOTE: not spending my time permuting for now
    self.bsize = args.batch_size
    self.shuffle = args.shuffle
    self.jsonfolder = jsonfolder--..'/jsons'
    self.outfolder = outfolder -- save stuff to here.

    -- here you can also include have world parameters
    print(jsonfolder)
    print(outfolder)
    if not(string.find(self.jsonfolder, 'tower') == nil) then
        self.maxwinsize = args.maxwinsize_long
    else
        self.maxwinsize = args.maxwinsize
    end

    if not(string.find(self.jsonfolder, '_dras_') == nil) or 
        not(string.find(self.jsonfolder, '_dras3_') == nil) then
        self.obj_sizes = args.drastic_object_sizes
    else 
        self.obj_sizes = args.object_sizes
    end

    return self
end

-- focus: (bsize, num_past, obj_dim)
-- context: ()
function data_process.k_nearest_context(focus, context, k)
    local bsize, num_past, obj_dim = context:size(1), context:size(3), context:size(4)

    -- euc_dist is a table of num_context entries of (bsize)
    -- table size is num_context
    local ed = data_process.get_euc_dist(focus:clone(), context:clone())  -- (bsize, num_context)  -- good

    -- for each example in bsize, you want to sort num_context and gets indices
    local k = math.min(12, ed:size(2))
    local closest, closest_indices = torch.topk(ed, k) -- get 12 closests

    -- here you can just sort closest_indices
    closest_indices = torch.sort(closest_indices)  -- sort in the original order they were presented

    -- print(context:size())

    local expand_size = torch.LongStorage{bsize,k,num_past,obj_dim}
    local new_context = context:clone():gather(2,torch.expand(closest_indices:view(mp.batch_size,k,1,1),expand_size))

    -- local new_context = {}
    -- for ex = 1, closest_indices:size(1) do  -- go through the batch
    --     local contexts_for_ex = context[{{ex}}]:index(2, closest_indices[ex])  -- (bsize, 12, num_past, obj_dim)
    --     table.insert(new_context, contexts_for_ex:clone())

    --     -- good up to here
    -- end
    -- new_context = torch.cat(new_context,1) -- good

    assert(new_context:size(1) == bsize and new_context:size(2) <= 12 and new_context:size(3) == num_past and new_context:size(4) == obj_dim)
    return new_context, closest_indices
end

-- good
function data_process.get_euc_dist(focus, context, t)
    local num_context = context:size(2)
    local t = t or -1  -- default use last timestep
    local px, py = config_args.rsi.px, config_args.rsi.py

    local this_pos = focus[{{},{t},{px, py}}]
    local context_pos = torch.squeeze(context[{{},{},{t},{px, py}}],3)

    local euc_dists = torch.squeeze(compute_euc_dist(this_pos:repeatTensor(1,num_context,1), context_pos),3) -- (bsize, num_context)
    return euc_dists
end



function data_process.relative_pair(past, future, relative_to_absolute)
    -- rta: relative to absolute, otherwise we are doing absolute to relative

    -- TODO: use config args for this!
    if relative_to_absolute then
        future[{{},{},{1,6}}] = future[{{},{},{1,6}}] + past[{{},{-1},{1,6}}]:expandAs(future[{{},{},{1,6}}])
    else
        future[{{},{},{1,6}}] = future[{{},{},{1,6}}] - past[{{},{-1},{1,6}}]:expandAs(future[{{},{},{1,6}}])
    end
    return future
end


-- (num_examples, num_objects, timestesp, a, av)
-- theta = theta-2pi if pi < theta < 2*pi
-- transforms [0, 2pi)  --> (-pi, pi]
function data_process:wrap_pi(angles)
    local angles = angles:clone()
    local wrap_mask = angles:gt(math.pi)  -- add this to angles
    local wrapped = torch.add(angles, -2*math.pi, wrap_mask:float())
    return wrapped
end

-- (num_examples, num_objects, timestesp, a, av)
-- theta = theta+2pi if -pi < theta < pi
-- transforms [-pi, pi] --> [0, 2pi]
function data_process:wrap_2pi(angles)
    local angles = angles:clone()    local wrap_mask = angles:lt(0)  -- add this to angles
    local wrapped = torch.add(angles, 2*math.pi, wrap_mask:float())
    return wrapped
end

-- trajectories: (num_examples, num_objects, timesteps, [px, py, vx, vy, mass])
function data_process:normalize(unnormalized_trajectories)
    normalized = unnormalized_trajectories:clone()

    local px, py, vx, vy = self.rsi.px, self.rsi.py, self.rsi.vx, self.rsi.vy
    local a, av = self.rsi.a, self.rsi.av

    -- normalize position
    normalized[{{},{},{},{px,py}}] = normalized[{{},{},{},{px,py}}]/self.pnc

    -- normalize velocity
    normalized[{{},{},{},{vx,vy}}] = normalized[{{},{},{},{vx,vy}}]/self.vnc

    -- transforms [0, 2pi]  --> [-pi, pi]
    normalized[{{},{},{},{a, av}}] = self:wrap_pi(normalized[{{},{},{},{a, av}}])


    -- normalize angle and angular velocity (assumes they are together)
    normalized[{{},{},{},{a, av}}] = normalized[{{},{},{},{a, av}}]/self.anc

    return normalized
end



function data_process:unnormalize(normalized_trajectories)
    unnormalized = normalized_trajectories:clone()

    local px, py, vx, vy = self.rsi.px, self.rsi.py, self.rsi.vx, self.rsi.vy
    local a, av = self.rsi.a, self.rsi.av

    -- normalize position
    unnormalized[{{},{},{},{px,py}}] = unnormalized[{{},{},{},{px,py}}]*self.pnc

    -- normalize velocity
    unnormalized[{{},{},{},{vx,vy}}] = unnormalized[{{},{},{},{vx,vy}}]*self.vnc

    -- normalize angle and angular velocity (assumes they are together)
    unnormalized[{{},{},{},{a, av}}] = unnormalized[{{},{},{},{a, av}}]*self.anc

    -- transforms [-pi, pi] --> [0, 2pi]
    unnormalized[{{},{},{},{a}}] = self:wrap_2pi(unnormalized[{{},{},{},{a}}])  -- no need to do this for av because it is between 0 and pi

    return unnormalized
end


function data_process:num2onehot(value, categories)
    return num2onehot(value, categories)
end

function data_process:onehot2num(onehot, categories)
    return onehot2num(onehot, categories)
end


function data_process:num2onehotall(selected, categories)
    return num2onehotall(selected, categories, false)
end


function data_process:onehot2numall(onehot_selected, categories)
    return onehot2numall(onehot_selected, categories, false)
end


function data_process:properties2onehotall(trajectories)  -- (num_ex, num_obj, num_steps, obj_dim)
    -- first, split 
    local before = trajectories[{{},{},{},{self.rsi.px, self.rsi.m-1}}]:clone()
    local masses = trajectories[{{},{},{},{self.rsi.m}}]:clone()
    local objtypes = trajectories[{{},{},{},{self.rsi.oid}}]:clone()
    local obj_sizes = trajectories[{{},{},{},{self.rsi.os}}]:clone()
    local gravity = trajectories[{{},{},{},{self.rsi.g}}]:clone()
    local friction = trajectories[{{},{},{},{self.rsi.f}}]:clone()
    local pairwise = trajectories[{{},{},{},{self.rsi.p}}]:clone()

    -- next, convert all to onehot
    masses = self:num2onehotall(masses, self.masses)  -- good
    objtypes = self:num2onehotall(objtypes, self.oid_ids)  -- good
    obj_sizes = self:num2onehotall(obj_sizes, self.obj_sizes)  -- good
    gravity = self:num2onehotall(gravity, self.boolean)  -- good
    friction = self:num2onehotall(friction, self.boolean)  -- good
    pairwise = self:num2onehotall(pairwise, self.boolean)  -- good

    -- last, rejoin
    local propertiesonehot = {masses, objtypes, obj_sizes,
                              gravity, friction, pairwise}
    local trajectoriesonehot = torch.cat({before, unpack(propertiesonehot)}, 4)  -- good
    return trajectoriesonehot
end

function data_process:onehot2propertiesall(trajectoriesonehot)
    -- first split
    local before = trajectoriesonehot[{{},{},{},{self.si.px, self.si.m[1]-1}}]:clone()
    local onehot_masses = trajectoriesonehot[{{},{},{},{unpack(self.si.m)}}]:clone()
    local onehot_objtypes = trajectoriesonehot[{{},{},{},{unpack(self.si.oid)}}]:clone()
    local onehot_obj_sizes = trajectoriesonehot[{{},{},{},{unpack(self.si.os)}}]:clone()
    local onehot_gravity = trajectoriesonehot[{{},{},{},{unpack(self.si.g)}}]:clone()
    local onehot_friction = trajectoriesonehot[{{},{},{},{unpack(self.si.f)}}]:clone()
    local onehot_pairwise = trajectoriesonehot[{{},{},{},{unpack(self.si.p)}}]:clone()

    -- next convert all to num
    masses = self:onehot2numall(onehot_masses, self.masses)
    objtypes = self:onehot2numall(onehot_objtypes, self.oid_ids)
    obj_sizes = self:onehot2numall(onehot_obj_sizes, self.obj_sizes)
    gravity = self:onehot2numall(onehot_gravity, self.boolean)
    friction = self:onehot2numall(onehot_friction, self.boolean)
    pairwise = self:onehot2numall(onehot_pairwise, self.boolean)

    -- last rejoin
    local propertiesnum = {masses, objtypes, obj_sizes,
                           gravity, friction, pairwise}
    local trajectories = torch.cat({before, unpack(propertiesnum)}, 4) 
    return trajectories
end


--[[ Expands the number of examples per batch to have an example per particle
    Input: unfactorized: (num_samples x num_obj x windowsize x 8)
    Output:
        {
            focus: (num_samples*num_obj, num_steps, obj_dim)
            context: (num_samples*num_obj x (num_obj-1) x num_steps x 8) or {}
        }
--]]
function data_process:expand_for_each_object(unfactorized)
    local num_samples, num_obj, num_steps, object_dim = unpack(torch.totable(unfactorized:size()))
    local focus = {}
    local context = {}
    local ball_index = self.si.oid[1]
    local obstacle_index = self.si.oid[1]+1
    local block_index = self.si.oid[2]

    local obj_index
    if not(string.find(self.jsonfolder, 'balls') == nil) or 
            not(string.find(self.jsonfolder, 'mixed') == nil) or 
            not(string.find(self.jsonfolder, 'invisible') == nil) or 
            not(string.find(self.jsonfolder, 'walls') == nil) then
        obj_index = self.si.oid[1]
    elseif not(string.find(self.jsonfolder, 'tower') == nil) then
        obj_index = self.si.oid[2]
    else
        assert(false, 'unknown focus object type')
    end

    if num_obj > 1 then
        for i=1,num_obj do  -- this is doing it in transpose order
            -- some objects will be balls, some obstacles, some invisible.
            -- since we are iterating through all the object indicies, here we just have to find the balls. Then we find the context accordingly.
            local focus_obj_mask = torch.squeeze(unfactorized[{{},{i},{1},{obj_index}}]:eq(1)) -- (num_samples)  -- we are only taking the first timestep because all timesteps are the same
            local num_selected = focus_obj_mask:sum()
            local focus_obj_indices = focus_obj_mask:nonzero()

            -- the examples of unfactorized where object i is a ball
            if focus_obj_indices:nElement() > 0 then  -- only construct examples if there are examples to construct.
                focus_obj_indices = torch.squeeze(focus_obj_indices,2)
                local selected_samples = unfactorized:clone():index(1,focus_obj_indices)  -- (num_selected, num_obj, num_steps, object_dim)  -- unnecessary to clone

                -- now find the focus object
                local this = torch.squeeze(selected_samples[{{},{i},{},{}}],2)

                -- now get the context objects
                local others
                if i == 1 then
                    others = selected_samples[{{},{i+1,-1},{},{}}]
                elseif i == num_obj then
                    others = selected_samples[{{},{1,i-1},{},{}}]
                else
                    others = torch.cat(selected_samples[{{},{1,i-1},{},{}}],
                                selected_samples[{{},{i+1,-1},{},{}}], 2)  -- leave this particle out (num_samples x (num_obj-1) x windowsize x object_dim)
                end

                assert(this:size()[1] == others:size()[1])
                table.insert(focus, this) -- good
                table.insert(context, others) -- good
            end
        end
    else
        -- make sure it is a ball
        assert(torch.squeeze(unfactorized[{{},{i},{1},{obj_index}}]:eq(1)):sum()==num_samples)
        local this = torch.squeeze(unfactorized[{{},{i},{},{}}],2)
        table.insert(focus, this)  -- (num_samples x num_steps x objdim)
        table.insert(context, torch.zeros(num_samples,1,num_steps,object_dim)) -- if just one object, then context is just zeross
    end

    focus = torch.cat(focus,1)  -- concatenate along batch dimension
    context = torch.cat(context,1)

    return focus, context
end


-- we also should have a method that divides the focus and context into past and future
-- this assumes we are predicting for everybody
function data_process:condense(focus, context)
    -- duplicates may exist, they may not because each object gets a chance to a focus object
    -- so the same set of trajectories would appear num_obj times
    focus = unsqueeze(focus, 2)
    -- TODO_lowpriority get rid of duplicates!
    return torch.cat({focus, context},2)
end

-- data:
function data_process:split2batches(data, truncate)
    print(data:size())
    local num_examples = data:size(1)
    -- here you should split through time

    local num_chunks = math.ceil(num_examples/self.bsize)
    print('Splitting '..num_examples..' examples into '..num_chunks..
            ' batches of size at most '..self.bsize)
    local result = data:clone():split(self.bsize,1)
    print(result)
    if truncate then
        if not(result[#result]:size(1) == self.bsize) then
            print('Last element not equal to self.bsize. Going to take that out.')
            print(result[#result]:size())
        end
        result = plt.sub(result, 1, #result-1)
        print(result)
    end
    return result
end


-- train-val-test: 70-15-15 split
function data_process:split_datasets_sizes(num_examples)
    assert(num_examples%1==0)
    local num_test = math.floor(num_examples * 0.15)
    local num_val = num_test
    local num_train = num_examples - 2*num_test
    -- if num_val == 0 then assert(false, 'valset and testset sizes are 0!') end
    return num_train, num_val, num_test
end


function data_process:save_batches(datasets, savefolder)
    if not paths.dirp(savefolder) then paths.mkdir(savefolder) end
    for k,v in pairs(datasets) do
        local dataset_folder = savefolder..'/'..k
        if not paths.dirp(dataset_folder) then paths.mkdir(dataset_folder) end
        print('Saving',k)
        for i=1,#v do
            xlua.progress(i,#v)
            local batch_file = dataset_folder..'/batch'..i
            torch.save(batch_file,v[i])
        end
    end
end

-- rejection sampling
function data_process:sample_dataset_id(dataset_ids, counters, limits)
    local dataset_id = math.ceil(torch.rand(1)[1]*#dataset_ids)
    while ((function ()
                local dataset_name = dataset_ids[dataset_id]
                if counters[dataset_name] >= limits[dataset_name] then
                    return true
                else return false end
            end)()) do
        dataset_id = math.ceil(torch.rand(1)[1]*#dataset_ids)
    end
    return dataset_id
end

function data_process:check_overflow(counters, limits)
    local buffer = 0
    for k,v in pairs(counters) do
        if counters[k] > limits[k] then
            return -1
        else
            buffer = buffer + limits[k] - counters[k]
        end
    end
    return buffer
end

function data_process:sample_save_single_batch(batch, dataset_ids, counters, limits)
    local dataset_id = self:sample_dataset_id(dataset_ids, counters, limits)
    counters[dataset_ids[dataset_id]] = counters[dataset_ids[dataset_id]] + 1

    -- save
    local dataset_folder = self.outfolder..'/'..dataset_ids[dataset_id]
    if not paths.dirp(dataset_folder) then paths.mkdir(dataset_folder) end  -- really redundant, should move this out
    local batch_file = dataset_folder..'/batch'..counters[dataset_ids[dataset_id]]
    print('Saving to '..batch_file)
    assert(not(paths.filep(batch_file)))
    torch.save(batch_file,batch)
    return counters
end

function data_process:iter_files_ordered(folder)
    local files = {}
    for f in paths.iterfiles(folder) do
        table.insert(files, f) -- good
    end
    table.sort(files)  -- mutates files
    return files
end

-- basically expands for each object first and counts the number of examples
-- if all balls, then the num_examples = total_samples*num_obj
-- this implementation depends on how expand_for_each_object is defined.
-- works
function data_process:count_examples(jsonfolder)
    local ordered_files = self:iter_files_ordered(jsonfolder)
    local oid_index = self.rsi.oid
    local obj_id
    if not(string.find(self.jsonfolder, 'balls') == nil) or 
            not(string.find(self.jsonfolder, 'mixed') == nil) or 
            not(string.find(self.jsonfolder, 'invisible') == nil) or 
            not(string.find(self.jsonfolder, 'walls') == nil) then
        obj_id = self.oid_ids[1]
    elseif not(string.find(self.jsonfolder, 'tower') == nil) then
        obj_id = self.oid_ids[3]
    else
        assert(false, 'unknown focus object type')
    end
    local num_examples = 0
    for _, jsonfile in pairs(ordered_files) do
        local data = load_data_json(paths.concat(jsonfolder,jsonfile))  -- (num_examples, num_obj, num_steps, object_raw_dim)
        local num_samples, num_obj, num_steps, object_dim = unpack(torch.totable(data:size()))

        -- print(num_samples, num_obj, num_steps, object_dim)
        -- assert(false)
        -- now count where there are balls (also works for tower)
        if num_obj > 1 then
            for i=1,num_obj do
                local ball_mask = torch.squeeze(data[{{},{i},{1},{oid_index}}]:eq(obj_id)) -- (num_samples)  -- we are only taking the first timestep because all timesteps are the same
                local num_selected = ball_mask:sum()
                num_examples = num_examples + num_selected
                print(num_selected..' examples with focus object in '..jsonfile)
            end
        else
            assert(torch.squeeze(unfactorized[{{},{i},{1},{oid_index}}]:eq(obj_id)):sum()==num_samples)
            num_examples = num_examples + num_samples
        end
    end
    collectgarbage()
    return num_examples
end

-- this sampling scheme is pretty complex, but it is random
-- if max_iters_per_json is a multiple of batch_size, then it should be fine
function data_process:create_datasets_batches()
    -- set up
    local flags = pls.split(string.gsub(self.jsonfolder,'/jsons',''), '_')
    local total_samples = tonumber(extract_flag(flags, 'ex'))  -- this is the number of trajectories
    local num_steps = tonumber(extract_flag(flags, 't'))
    local num_obj

    if not(string.find(self.jsonfolder, 'walls') == nil) then
        -- U: 33
        -- L: 30
        -- O: 30
        -- I: 32   
        if not(string.find(self.jsonfolder, '_wO') == nil) or not(string.find(self.jsonfolder, '_wL') == nil) then
            num_obj = 30
        elseif not(string.find(self.jsonfolder, '_wU') == nil) then
            num_obj = 33
        elseif not(string.find(self.jsonfolder, '_wI') == nil) then
            num_obj = 32
        else
            assert(false, 'unknown wall type')
        end
    else
        num_obj = tonumber(extract_flag(flags, 'n'))
    end
    print('num obj', num_obj)


    print('Counting Examples')
    local num_examples = self:count_examples(self.jsonfolder)
    if not(string.find(self.jsonfolder, 'tower') == nil) or not(string.find(self.jsonfolder, 'balls') == nil) then
        assert(num_examples == total_samples*num_obj)
    end
    print('Total number of examples: '..num_examples)

    local num_batches = math.floor(num_examples/self.bsize)
    print('Number of batches: '..num_batches..' with batch size '..self.bsize)
    local num_train, num_val, num_test = self:split_datasets_sizes(num_batches)
    print('train: '..num_train..' val: '..num_val..' test: '..num_test)

    local counters = {trainset=0, valset=0, testset=0}
    local dataset_ids = {'trainset', 'valset', 'testset'}
    local limits = {trainset=num_train, valset=num_val, testset=num_test}

    -- now, let's implement the queue
    local leftover_examples = {}

    ----------------------------------------------------------
    -- it's a good thing I don't need to do leftover examples

    -- local counters = {trainset=3400, valset=750, testset=750}
    -- -- now I need to figure out which jsons have not been seen yet.
    -- local count_logfile = 'tower_n10_t120_ex25000_rd_count.txt'
    -- local seen_jsons = {}
    -- for line in io.lines(count_logfile) do table.insert(seen_jsons, line) end

    -- TODODO
    local ordered_files = self:iter_files_ordered(self.jsonfolder)
    for _, jsonfile in pairs(ordered_files) do 
    --      if not(isin(jsonfile, jsons)) do
    --          then do the stuff
    ----------------------------------------------------------


    -- for jsonfile in paths.iterfiles(self.jsonfolder) do  -- order doesn't matter

        ----------------------------------------------------------
        -- if not(isin(jsonfile, seen_jsons)) then

        ----------------------------------------------------------

       -- note that this may not all be the same batch size! They will even out at the end though
       local new_batches = self:json2batches(paths.concat(self.jsonfolder,jsonfile))
       print('new batches')
       print(new_batches)
       for _, batch in pairs(new_batches) do
           assert(self:check_overflow(counters, limits) >= 0)
           -- check to see if this batch is of batch_size
           if batch[1]:size(1) < self.bsize then
               table.insert(leftover_examples, batch)  -- good
               print('leftover examples')
               print(leftover_examples)
           else
               -- sample which dataset you should save it in
                counters = self:sample_save_single_batch(batch, dataset_ids, counters, limits)
            end
           collectgarbage()
       end
       ----------------------------------------------------------
       -- end
       ----------------------------------------------------------
    end

    -- now concatenate all the leftover_batches. They had better be a multiple of self.bsize
    leftover_examples = join_table_of_tables(leftover_examples)
    -- leftover_examples = join_table_of_tables({unpack(leftover_examples), unpack(leftover_examples)})  -- for debugging

    print('Merged leftover examples:')
    print(leftover_examples)
    if #leftover_examples > 0 then
        assert(leftover_examples[1]:size(1)==leftover_examples[2]:size(1))  -- check that focus and context have same number of batches
        -- assert(leftover_examples[1]:size(1) % self.bsize == 0)  -- this is taken care of by our truncation = true below
        -- assert(self:check_overflow(counters, limits)*self.bsize == leftover_examples[1]:size(1)) -- we have exactly enough examples to fill the dataset quotas
        local leftover_batches = self:split2batchesall(leftover_examples[1], leftover_examples[2], true)  -- guaranteed to output batches of self.bsize
        assert(self:check_overflow(counters, limits) == #leftover_batches)  -- we have exactly enough batches left to fill the dataset quotas
        print('Saving leftover_batches')
        print(leftover_batches)
        for _, batch in pairs(leftover_batches) do
            assert(self:check_overflow(counters, limits) >= 0)
            counters = self:sample_save_single_batch(batch, dataset_ids, counters, limits)
        end
    end
end

-- perfomrs split2batches on both focus and context and then merges result
-- focus (num_samples*num_obj, num_steps, obj_dim)
-- context (num_samples*num_obj, num_obj-1, num_steps, obj_dim)
function data_process:split2batchesall(focus, context, truncate)
    local focus_batches = self:split2batches(focus, truncate)
    local context_batches = self:split2batches(context, truncate)
    local all_batches = {}
    for b=1,#focus_batches do
        table.insert(all_batches, {focus_batches[b], context_batches[b]}) -- good
    end
    return all_batches
end

function data_process:json2batches(jsonfile)
    local data = load_data_json(jsonfile)
    assert(data:size(3) == self.maxwinsize)
    data = self:normalize(data)  -- good
    data = self:properties2onehotall(data)  -- good
    local focus, context = self:expand_for_each_object(data)
    return self:split2batchesall(focus, context)
end

-- this method converts torch back to json file
-- input: (bsize, num_obj, steps, dim) for focus and context, with onehotmass, and normalized
-- batch size can be one
-- assume that the trajectories are not sliced into past and future for now
function data_process:record_trajectories(batch, config, jsonfile)
    -- now I have to combine focus and context and remove duplicates?
    local trajectories = self:condense(unpack(batch))
    local trajectories = self:onehot2propertiesall(trajectories)
    local unnormalized = self:unnormalize(trajectories)
    local batch_table = data2table(unnormalized)
    json.save(jsonfile, {trajectories=batch_table,config=config})
end

return data_process
