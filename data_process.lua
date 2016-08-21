-- normalize: requires normalizing constants

-- unnormalize

-- permute (we could possibly use this)

-- one hot (masses)

-- subsample range (we could possibly use this)

-- split into train, validation, test

-- group into batches?

-- fields: relative, object_dim

-- also need to split past/future? Or should that be in the computation?

-- note that these functions do NOT mutate state. They just reference fields
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
    -- self.maxwinsize = args.maxwinsize
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

    if not(string.find(self.jsonfolder, '_dras_') == nil) then
        self.obj_sizes = args.drastic_object_sizes
    else 
        self.obj_sizes = args.object_sizes
    end

    return self
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
-- transforms [0, 2pi]  --> [-pi, pi]
function data_process:wrap_pi(angles)
    local angles = angles:clone()
    -- angles = torch.clamp(angle, 0, 2*math.pi)  -- just in case

    local wrap_mask = angles:gt(math.pi)  -- add this to angles
    -- print(wrap_mask)
    local wrapped = torch.add(angles, -2*math.pi, wrap_mask:float())
    return wrapped
end

-- (num_examples, num_objects, timestesp, a, av)
-- theta = theta+2pi if -pi < theta < pi
-- transforms [-pi, pi] --> [0, 2pi]

function data_process:wrap_2pi(angles)
    local angles = angles:clone()
    -- angles = torch.clamp(angle, -math.pi, math.pi)  -- just in case

    local wrap_mask = angles:lt(0)  -- add this to angles
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
    unnormalized[{{},{},{},{a, av}}] = self:wrap_2pi(unnormalized[{{},{},{},{a, av}}])

    return unnormalized
end


function data_process:num2onehot(value, categories)
    local index = torch.find(torch.Tensor(categories), value)[1]
    -- print(value)
    -- print(categories)
    assert(not(index == nil))
    local onehot = torch.zeros(#categories)
    onehot[{{index}}]:fill(1)  -- will throw an error if index == nil
    return onehot
end

function data_process:onehot2num(onehot, categories)
    assert(onehot:sum() == 1 and #torch.find(onehot, 1) == 1)
    return categories[torch.find(onehot, 1)[1]]
end


function data_process:num2onehotall(selected, categories)
    local num_ex = selected:size(1)
    local num_obj = selected:size(2)
    local num_steps = selected:size(3)

    -- expand
    selected = torch.repeatTensor(selected, 1, 1, 1, #categories)  -- I just want to tile on the last dimension
    selected:resize(num_ex*num_obj*num_steps, #categories)

    for row=1,selected:size(1) do
        selected[{{row}}] = self:num2onehot(selected[{{row},{1}}]:sum(), categories)
    end
    selected:resize(num_ex, num_obj, num_steps, #categories)
    return selected
end


function data_process:onehot2numall(onehot_selected, categories)
    local num_ex = onehot_selected:size(1)
    local num_obj = onehot_selected:size(2)
    local num_steps = onehot_selected:size(3)

    local selected = torch.zeros(num_ex*num_obj*num_steps, 1)  -- this is not cuda-ed!
    onehot_selected:resize(num_ex*num_obj*num_steps, #categories)

    for row=1,onehot_selected:size(1) do
        selected[{{row}}] = self:onehot2num(torch.squeeze(onehot_selected[{{row}}]), categories)
    end
    selected:resize(num_ex, num_obj, num_steps, 1)
    return selected
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
            not(string.find(self.jsonfolder, 'invisible') == nil)then
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
            -- print(focus_obj_indices:nElement())
            if focus_obj_indices:nElement() > 0 then  -- only construct examples if there are examples to construct.
                focus_obj_indices = torch.squeeze(focus_obj_indices,2)
                local selected_samples = unfactorized:clone():index(1,focus_obj_indices)  -- (num_selected, num_obj, num_steps, object_dim)  -- unnecessary to clone
                -- sanity check
                -- print(torch.squeeze(unfactorized[{{89,94},{i},{1},self.si.oid}]))
                -- print(torch.squeeze(unfactorized[{{89,94},{i},{1},{}}]))
                -- print(torch.squeeze(selected_samples[{{},{i},{1}}]))

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
                table.insert(focus, this)
                table.insert(context, others)
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

    -- print(focus)
    -- print(focus:norm())
    -- assert(false)

    -- TODO! Do I have to crop?
    return focus, context
end



-- --[[ Expands the number of examples per batch to have an example per particle
--     Input: unfactorized: (num_samples x num_obj x windowsize x 8)
--     Output:
--         {
--             focus: (num_samples*num_obj, num_steps, obj_dim)
--             context: (num_samples*num_obj x (num_obj-1) x num_steps x 8) or {}
--         }
-- -- --]]
-- function data_process:expand_for_each_object(unfactorized)
--     local num_samples, num_obj, _, _ = unpack(torch.totable(unfactorized:size()))
--     local focus = {}
--     local context = {}
--     if num_obj > 1 then
--         for i=1,num_obj do  -- this is doing it in transpose order
--             local this = unfactorized[{{},{i},{},{}}]  --all of the particles here should be the same  -- NOT NECESSARILY TRUE!
--             local ball_index = self.si.oid[1]
--             local obstacle_index = self.si.oid[1]+1
--             local block_index = self.si.oid[2]
--             if this[{{},{},{},{obstacle_index}}]:sum() == 0 or (not(string.find(self.jsonfolder, 'invisible') == nil) and this[{{},{},{},{block_index}}]:sum() == 0) then -- only do it if the particle is not stationary obstacle
--                 this = this:reshape(this:size(1), this:size(3), this:size(4))  -- (num_samples x windowsize x obj_dim); NOTE that resize gives the wrong answer!
--                 local other
--                 if i == 1 then
--                     other = unfactorized[{{},{i+1,-1},{},{}}]
--                 elseif i == num_obj then
--                     other = unfactorized[{{},{1,i-1},{},{}}]
--                 else
--                     other = torch.cat(unfactorized[{{},{1,i-1},{},{}}],
--                                 unfactorized[{{},{i+1,-1},{},{}}], 2)  -- leave this particle out (num_samples x (num_obj-1) x windowsize x 8)
--                 end

--                 -- TODOlowpriority should permute here
--                 assert(this:size()[1] == other:size()[1])
--                 focus[#focus+1] = this
--                 context[#context+1] = other
--             end
--         end
--     else
--         local this = unfactorized[{{},{i},{},{}}]
--         focus[#focus+1] = torch.squeeze(this,2) -- (num_samples x windowsize x objdim)
--     end

--     focus = torch.cat(focus,1)  -- concatenate along batch dimension

--     -- make context into Torch tensor if more than one particle. Otherwise {}
--     if next(context) then
--         context = torch.cat(context,1)
--         assert(focus:size(1) == context:size(1))
--         assert(context:size(2) == num_obj-1)
--     end
--     -- print(focus)
--     -- print(focus:norm())
--     -- assert(false)
--     return focus, context
-- end


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


function data_process:split2datasets(examples)
    local num_train, num_val, num_test = self:split_datasets_sizes(#examples)

    local test = {}
    local val = {}
    local train = {}

    -- shuffle examples
    local ridxs = torch.randperm(#examples)
    print('Splitting datsets: '..string.format('%2d train %2d val %2d test',
                                                num_train, num_val, num_test))
    for i = 1, ridxs:size(1) do
        xlua.progress(i, ridxs:size(1))
        local batch = examples[ridxs[i]]
        if i <= num_train then
            table.insert(train, batch)
        elseif i <= num_train + num_val then
            table.insert(val, batch)
        else
            table.insert(test, batch)
        end
    end
    return {trainset=train, valset=val, testset=test}
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
    -- Wrote to ../data/mixed_n6_t60_ex50000_rd/jsons/mixed_n6_t60_ex50000_rd_chksize100_188.json
    -- Wrote to ../data/mixed_n6_t60_ex50000_rd/jsons/mixed_n6_t60_ex50000_rd_chksize100_189.json
    -- Wrote to ../data/mixed_n6_t60_ex50000_rd/jsons/mixed_n6_t60_ex50000_rd_chksize100_190.json
    -- Wrote to ../data/mixed_n6_t60_ex50000_rd/jsons/mixed_n6_t60_ex50000_rd_chksize100_191.json
    -- Wrote to ../data/mixed_n6_t60_ex50000_rd/jsons/mixed_n6_t60_ex50000_rd_chksize100_192.json
    local files = {}
    for f in paths.iterfiles(folder) do
        table.insert(files, f)
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
            not(string.find(self.jsonfolder, 'invisible') == nil)then
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
    local num_obj = tonumber(extract_flag(flags, 'n'))
    local num_steps = tonumber(extract_flag(flags, 't'))

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


       local new_batches = self:json2batches(paths.concat(self.jsonfolder,jsonfile))  -- note that this may not all be the same batch size! They will even out at the end though
       print('new batches')
       print(new_batches)
       for _, batch in pairs(new_batches) do
           assert(self:check_overflow(counters, limits) >= 0)
           -- check to see if this batch is of batch_size
           if batch[1]:size(1) < self.bsize then
               table.insert(leftover_examples, batch)
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
    else
        -- verify that all batches have been saved TODOlowpriority
        -- self.num_batches = tonumber(sys.execute("ls -1 " .. self.savefolder .. "/ | wc -l"))
    end
    -- assert(false)
end

-- perfomrs split2batches on both focus and context and then merges result
-- focus (num_samples*num_obj, num_steps, obj_dim)
-- context (num_samples*num_obj, num_obj-1, num_steps, obj_dim)
function data_process:split2batchesall(focus, context, truncate)
    local focus_batches = self:split2batches(focus, truncate)
    local context_batches = self:split2batches(context, truncate)
    local all_batches = {}
    for b=1,#focus_batches do
        table.insert(all_batches, {focus_batches[b], context_batches[b]})
    end
    return all_batches
end

function data_process:json2batches(jsonfile)
    local data = load_data_json(jsonfile)
    -- print(data)
    assert(data:size(3) == self.maxwinsize)
    data = self:normalize(data)  -- good
    data = self:properties2onehotall(data)  -- good
    local focus, context = self:expand_for_each_object(data)
    return self:split2batchesall(focus, context)
end

-- save datasets
function data_process:create_datasets()
    -- each example is a (focus, context) pair
    local json_file = self.jsonfolder --'/Users/MichaelChang/Documents/Researchlink/SuperUROP/Code/physics_worlds/tower.json'
    local all_batches = self:json2batches(jsonfile)
    local datasets = self:split2datasets(all_batches)
    self:save_batches(datasets, self.outfolder)
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


-- local args = require 'config'
--
-- dp = data_process.create(args)
-- dp:create_datasets()

-- now test if it can record
-- batch1 = torch.load('debug/train/batch1')
-- dp:record_trajectories(batch1, 'batch1.json')
