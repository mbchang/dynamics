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
    self.rsi = args.rsi -- {px: 1, py: 2, vx: 3, vy: 4, m: 5, oid: 6}
    self.si = args.si -- {px: {1}, py: {2}, vx: {3}, vy: {4}, m: {5,8}, oid: {9}}
    self.permute_context = args.permute_context  -- bool: if True will expand the dataset, False won't NOTE: not spending my time permuting for now
    self.bsize = args.batch_size
    self.shuffle = args.shuffle
    self.jsonfolder = jsonfolder--..'/jsons'
    self.outfolder = outfolder -- save stuff to here.

    -- here you can also include have world parameters
    -- not that object id includes whether it is stationary or not

    return self
end

-- trajectories: (num_examples, num_objects, timesteps, [px, py, vx, vy, mass])
function data_process:normalize(unnormalized_trajectories)
    normalized = unnormalized_trajectories:clone()

    local px, py, vx, vy = self.rsi.px, self.rsi.py, self.rsi.vx, self.rsi.vy
    local a, av = self.rsi.a, self.rsi.v

    -- normalize position
    normalized[{{},{},{},{px,py}}] = normalized[{{},{},{},{px,py}}]/self.pnc

    -- normalize velocity
    normalized[{{},{},{},{vx,vy}}] = normalized[{{},{},{},{vx,vy}}]/self.vnc

    -- normalize angle and angular velocity (assumes they are together)
    normalized[{{},{},{},{a, av}}] = normalized[{{},{},{},{a, av}}]/self.anc

    return normalized
end

function data_process:unnormalize(normalized_trajectories)
    unnormalized = normalized_trajectories:clone()

    local px, py, vx, vy = self.rsi.px, self.rsi.py, self.rsi.vx, self.rsi.vy
    local a, av = self.rsi.a, self.rsi.v

    -- normalize position
    unnormalized[{{},{},{},{px,py}}] = unnormalized[{{},{},{},{px,py}}]*self.pnc

    -- normalize velocity
    unnormalized[{{},{},{},{vx,vy}}] = unnormalized[{{},{},{},{vx,vy}}]*self.vnc

    -- normalize angle and angular velocity (assumes they are together)
    unnormalized[{{},{},{},{a, av}}] = unnormalized[{{},{},{},{a, av}}]*self.anc

    return unnormalized
end

function data_process:mass2onehot(mass)
    local index = torch.find(torch.Tensor(self.masses), mass)[1]
    assert(not(index == nil))
    local onehot = torch.zeros(#self.masses)
    onehot[{{index}}]:fill(1)  -- will throw an error if index == nil
    return onehot
end

function data_process:onehot2mass(onehot)
    assert(onehot:sum() == 1 and #torch.find(onehot, 1) == 1)
    return self.masses[torch.find(onehot, 1)[1]]
end

function data_process:mass2onehotall(trajectories)
    local before = trajectories[{{},{},{},{self.rsi.px, self.si.m[1]-1}}]:clone()
    local after = trajectories[{{},{},{},{self.rsi.m+1,-1}}]:clone()
    local masses = trajectories[{{},{},{},{self.rsi.m}}]:clone()

    local num_ex = masses:size(1)
    local num_obj = masses:size(2)
    local num_steps = masses:size(3)

    -- expand
    masses = torch.repeatTensor(masses, 1, 1, 1, #self.masses)  -- I just want to tile on the last dimension
    masses:resize(num_ex*num_obj*num_steps, #self.masses)

    for row=1,masses:size(1) do
        masses[{{row}}] = self:mass2onehot(masses[{{row},{1}}]:sum())
    end
    masses:resize(num_ex, num_obj, num_steps, #self.masses)

    -- join
    local trajectoriesonehot = torch.cat({before, masses, after}, 4)

    return trajectoriesonehot
end

-- Do I need a onehot2massall method?
function data_process:onehot2massall(trajectoriesonehot)
    local before = trajectoriesonehot[{{},{},{},{self.si.px, self.si.m[1]-1}}]:clone()
    local after = trajectoriesonehot[{{},{},{},{self.si.m[2]+1,-1}}]:clone()
    local onehot_masses = trajectoriesonehot[{{},{},{},{unpack(self.si.m)}}]:clone()

    local num_ex = onehot_masses:size(1)
    local num_obj = onehot_masses:size(2)
    local num_steps = onehot_masses:size(3)

    local masses = torch.zeros(num_ex*num_obj*num_steps, 1)  -- this is not cuda-ed!
    onehot_masses:resize(num_ex*num_obj*num_steps, #self.masses)

    for row=1,onehot_masses:size(1) do
        masses[{{row}}] = self:onehot2mass(torch.squeeze(onehot_masses[{{row}}]))
    end
    masses:resize(num_ex, num_obj, num_steps, 1)

    -- join
    local trajectories = torch.cat({before, masses, after}, 4)

    return trajectories
end

--[[ Expands the number of examples per batch to have an example per particle
    Input: unfactorized: (num_samples x num_obj x windowsize x 8)
    Output:
        {
            focus: (num_samples, windowsize, 8)
            context: (num_samples x (num_obj-1) x windowsize x 8) or {}
        }
--]]
function data_process:expand_for_each_object(unfactorized)
    local num_samples, num_obj, _, _ = unpack(torch.totable(unfactorized:size()))
    local focus = {}
    local context = {}
    if num_obj > 1 then
        for i=1,num_obj do  -- this is doing it in transpose order
            -- NOTE: the one-hot encoding has 4 values, and if the last value is 1 that means it is the stationary ball!
            local this = unfactorized[{{},{i},{},{}}]  --all of the particles here should be the same
            if this[{{},{},{},{-2}}]:sum() == 0 then -- only do it if the particle is not stationary
                this = this:reshape(this:size(1), this:size(3), this:size(4))  -- (num_samples x windowsize x 8); NOTE that resize gives the wrong answer!

                local other
                if i == 1 then
                    other = unfactorized[{{},{i+1,-1},{},{}}]
                elseif i == num_obj then
                    other = unfactorized[{{},{1,i-1},{},{}}]
                else
                    other = torch.cat(unfactorized[{{},{1,i-1},{},{}}],
                                unfactorized[{{},{i+1,-1},{},{}}], 2)  -- leave this particle out (num_samples x (num_obj-1) x windowsize x 8)
                end

                -- permute here
                assert(this:size()[1] == other:size()[1])
                -- this: (num_samples, winsize, 8)
                -- other: (num_samples, num_other_particles, winsize, 8)
                -- local num_other_particles = other:size(2)
                -- for j = 1, num_other_particles do
                --
                --     local permuted_other = torch.cat(permute(other),1)
                --     assert(permuted_other:size(1) == factorial(num_other_particles))
                --     for k = 1, factorial(num_other_particles) do
                --         focus[#focus+1] = this
                --     end
                --     context[#context+1] = permuted_other
                --
                --     -- focus[#focus+1] = this
                --     -- context[#context+1] = other
                -- end
                focus[#focus+1] = this
                context[#context+1] = other
            end
        end
    else
        local this = unfactorized[{{},{i},{},{}}]
        focus[#focus+1] = torch.squeeze(this,2) -- (num_samples x windowsize x 8)
    end

    focus = torch.cat(focus,1)  -- concatenate along batch dimension

    -- make context into Torch tensor if more than one particle. Otherwise {}
    if next(context) then
        context = torch.cat(context,1)
        assert(focus:size(1) == context:size(1))
        assert(context:size(2) == num_obj-1)
    end

    return focus, context
end


-- we also should have a method that divides the focus and context into past and future
-- this assumes we are predicting for everybody
function data_process:condense(focus, context)

    -- duplicates may exist, they may not
    focus = unsqueeze(focus, 2)
    -- TODO: get rid of duplicates!
    return torch.cat({focus, context},2)
end

function data_process:split2batches(data)
    local num_examples = data:size(1)
    -- assert(num_examples % self.bsize == 0)  -- TODO take care of this!
    local num_chunks = math.ceil(num_examples/self.bsize)
    print('Splitting '..num_examples..' examples into '..num_chunks..
            ' batches of size at most '..self.bsize)
    local result = data:clone():split(self.bsize,1)
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

-- function data_process:extract_flag(flags_list, delim)
--     local extract = plt.filter(flags_list, function(x) return pls.startswith(x, delim) end)
--     assert(#extract == 1)
--     return string.sub(extract[1], #delim+1)
-- end

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

-- this sampling scheme is pretty complex, but it is random
-- if max_iters_per_json is a multiple of batch_size, then it should be fine
function data_process:create_datasets_batches()
    --max_iters_per_json

    local flags = pls.split(string.gsub(self.jsonfolder,'/jsons',''), '_')
    local total_samples = tonumber(extract_flag(flags, 'ex'))
    local num_obj = tonumber(extract_flag(flags, 'n'))
    local num_steps = tonumber(extract_flag(flags, 't'))
    -- local total_samples = tonumber(self:extract_flag(flags, 'ex'))
    -- local num_obj = tonumber(self:extract_flag(flags, 'n'))
    -- local num_steps = tonumber(self:extract_flag(flags, 't'))
    local num_batches = total_samples*num_obj/self.bsize -- TODO: this can change based on on how you want to process num_steps!
    print(num_obj..' objects '..num_steps..' steps '..total_samples..
        ' samples yields '..num_batches..' batches')
    local num_train, num_val, num_test = self:split_datasets_sizes(num_batches)
    print('train: '..num_train..' val: '..num_val..' test: '..num_test)

    local counters = {trainset=0, valset=0, testset=0}
    local dataset_ids = {'trainset', 'valset', 'testset'}
    local limits = {trainset=num_train, valset=num_val, testset=num_test}

    -- now, let's implement the queue
    local leftover_examples = {}
    for jsonfile in paths.iterfiles(self.jsonfolder) do  -- order doesn't matter
       local new_batches = self:json2batches(paths.concat(self.jsonfolder,jsonfile))  -- note that this may not all be the same batch size! They will even out at the end though
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
    end
    -- now concatenate all the leftover_batches. They had better be a multiple of self.bsize
    leftover_examples = join_table_of_tables(leftover_examples)
    print('Merged leftover examples:')
    print(leftover_examples)
    if #leftover_examples > 0 then
        assert(leftover_examples[1]:size(1)==leftover_examples[2]:size(1))
        assert(leftover_examples[1]:size(1) % self.bsize == 0)
        assert(self:check_overflow(counters, limits)*self.bsize == leftover_examples[1]:size(1)) -- we have exactly enough examples to fill the dataset quotas
        local leftover_batches = self:split2batchesall(leftover_examples[1], leftover_examples[2])
        print('Saving leftover_batches')
        print(leftover_batches)
        for _, batch in pairs(leftover_batches) do
            assert(self:check_overflow(counters, limits) >= 0)
            counters = self:sample_save_single_batch(batch, dataset_ids, counters, limits)
        end
    else
        -- verify that all batches have been saved TODO
        -- self.num_batches = tonumber(sys.execute("ls -1 " .. self.savefolder .. "/ | wc -l"))
    end
end

-- perfomrs split2batches on both focus and context and then merges result
function data_process:split2batchesall(focus, context)
    local focus_batches = self:split2batches(focus)
    local context_batches = self:split2batches(context)
    local all_batches = {}
    for b=1,#focus_batches do
        table.insert(all_batches, {focus_batches[b], context_batches[b]})
    end
    return all_batches
end

function data_process:json2batches(jsonfile)
    local data = load_data_json(jsonfile)
    data = self:normalize(data)
    data = self:mass2onehotall(data)
    local focus, context = self:expand_for_each_object(data)-- TODO include object id in expand for each object
    return self:split2batchesall(focus, context)
end

-- save datasets
function data_process:create_datasets()
    -- each example is a (focus, context) pair
    local json_file = self.jsonfolder --'/Users/MichaelChang/Documents/Researchlink/SuperUROP/Code/physics_worlds/tower.json'
    local all_batches = self:json2batches(jsonfile)
    local datasets = self:split2datasets(all_batches)
    self:save_batches(datasets, self.outfolder)

    -- TODO: you should also save a sort of config file in the folder such that you can match the winsize and stuff in your dataloader, for example
end

-- this method converts torch back to json file
-- input: (bsize, num_obj, steps, dim) for focus and context, with onehotmass, and normalized
-- batch size can be one
-- assume that the trajectories are not sliced into past and future for now
function data_process:record_trajectories(batch, config, jsonfile)
    -- now I have to combine focus and context and remove duplicates?
    local trajectories = self:condense(unpack(batch))
    local trajectories = self:onehot2massall(trajectories)
    local unnormalized = self:unnormalize(trajectories)
    -- dump_data_json(unnormalized, jsonfile)
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
