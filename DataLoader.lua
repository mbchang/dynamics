-- data loader for object model

require 'metaparams'
require 'torch'
require 'math'
require 'image'
require 'lfs'
require 'torch'
require 'paths'
require 'hdf5'
require 'data_utils'
require 'torchx'
require 'utils'
local T = require 'pl.tablex'

if common_mp.cuda then require 'cutorch' end
if common_mp.cunn then require 'cunn' end

local dataloader = {}
dataloader.__index = dataloader

local object_dim = 8
local max_other_objects = 10
local all_worlds = {'worldm1', 'worldm2', 'worldm3', 'worldm4'}


--[[ Loads the dataset as a table of configurations

    Input
        .h5 file data
        The data for each configuration is spread out across 3 keys in the .h5 file
        Let <config> be the configuration name
            "<config>particles": (num_examples, num_particles, windowsize, [px, py, vx, vy, (onehot mass)])
            "<config>goos": (num_examples, num_goos, [left, top, right, bottom, (onehot goostrength)])
            "<config>mask": binary vector of length 5, trailing zeros are padding when #particles < 6

    Output
    {
        configuration:
            {
              particles : DoubleTensor - size: (num_examples x num_particles x windowsize x 5)
              goos : DoubleTensor - size: (num_examples x num_goos x 5)
              mask : DoubleTensor - size: 5
            }
    }

    The last dimension of particles is 8 because: [px, py, vx, vy, (onehot mass)]
    The last dimension of goos is 8 because: [left, top, right, bottom, (onehot goostrength)]
    The mask is dimension 8 because: our data has at most 6 particles -- ]]
function load_data(dataset_name, dataset_folder)
    local dataset_file = hdf5.open(dataset_folder .. '/' .. dataset_name, 'r')

    -- Get all keys: note they might not be in order though!
    local examples = {}
    local subkeys = {'particles', 'goos', 'mask'}  -- hardcoded
    for k,v in pairs(dataset_file:all()) do

        -- find the subkey of interest
        local this_subkey
        local example_key
        local counter = 0
        for sk, sv in pairs(subkeys) do
            if k:find(sv) then
                counter = counter + 1
                this_subkey = sv
                example_key = k:sub(0,k:find(sv)-1)
            end
        end
        assert(counter == 1)
        assert(this_subkey and example_key)

        if examples[example_key] then
            examples[example_key][this_subkey] = v
        else
            examples[example_key] = {}
            examples[example_key][this_subkey] = v
        end
    end
    return examples
end


function dataloader.create(dataset_name, dataset_folder, specified_configs, batch_size, curriculum, shuffle)
    --[[
        Input
            dataset_name: file containing data, like 'trainset'
            dataset_folder: folder containing the .h5 files
            shuffle: boolean


            What I want to be able to is to have a dataloader, that takes in parameters:
                - dataset_name?
                - shuffle
                - a table of configs (indexed by number, in order)
                - batch size

            Then when I do next_batch, it will go through appropriately.

            specified_configs = table of worlds or configs
    --]]
    assert(all_args_exist{dataset_name, dataset_folder, specified_configs,batch_size,curriculum,shuffle})

    local self = {}
    setmetatable(self, dataloader)

    ---------------------------------- Givens ----------------------------------
    self.dataset_name = dataset_name  -- string
    self.batch_size = batch_size
    self.object_dim = object_dim
    if curriculum then assert(not shuffle) end

    -------------------------------- Get Dataset -----------------------------
    self.dataset = load_data(dataset_name..'.h5', dataset_folder)  -- table of all the data
    self.configs = get_keys(self.dataset)  -- table of all keys

    ---------------------- Focus Dataset to Specification ----------------------
    if is_empty(specified_configs) then
        self.specified_configs = self.configs
    elseif contains_world(specified_configs) then
        self.specified_configs = get_all_specified_configs(specified_configs, T.deepcopy(self.configs))
    else
        self.specified_configs = specified_configs
    end
    assert(is_subset(self.specified_configs, self.configs))
    if not shuffle then topo_order(self.specified_configs) end
    self.num_configs = #self.specified_configs
    self.config_idxs = torch.range(1,self.num_configs)
    self.total_examples, self.num_batches, self.config_sizes = self:count_examples(self.specified_configs)  -- TODO count_examples will take in argument

    ----------------------- Initial values for iterator ------------------------
    self.batchlist = self:compute_batches()  -- TODO
    self.current_batch = 0

    ---------------------------------- Shuffle ---------------------------------
    if shuffle then
        self.batch_idxs = torch.randperm(self.num_batches)
    else
        self.batch_idxs = torch.range(1,self.num_batches)
    end

    collectgarbage()
    return self
end



--[[ Expands the number of examples per batch to have an example per particle
    Input: batch_particles: (num_samples x num_particles x windowsize x 8)
    Output:
        {
            this_particles: (num_samples, windowsize, 8)
            other_particles: (num_samples x (num_particles-1) x windowsize x 8) or {}
        }
--]]
function expand_for_each_particle(batch_particles)
    local num_samples, num_particles, windowsize = unpack(torch.totable(batch_particles:size()))
    local this_particles = {}
    local other_particles = {}
    local this_idxes = {}
    if num_particles > 1 then
        for i=1,num_particles do  -- this is doing it in transpose order
            local this = batch_particles[{{},{i},{},{}}]
            this = this:reshape(this:size(1), this:size(3), this:size(4))  -- (num_samples x windowsize x 8)
            -- this = this:resize(this:size(1), this:size(3), this:size(4))  -- (num_samples x windowsize x 8)  -- this is the issue! resize actually gives the wrong answer!
            this_particles[#this_particles+1] = this
            this_idxes[#this_idxes+1] = i

            local other
            if i == 1 then
                other = batch_particles[{{},{i+1,-1},{},{}}]
            elseif i == num_particles then
                other = batch_particles[{{},{1,i-1},{},{}}]
            else
                other = torch.cat(batch_particles[{{},{1,i-1},{},{}}],
                            batch_particles[{{},{i+1,-1},{},{}}], 2)  -- leave this particle out (num_samples x (num_particles-1) x windowsize x 8)
            end
            assert(this:size()[1] == other:size()[1])
            other_particles[#other_particles+1] = other
        end
    else
        local this = batch_particles[{{},{i},{},{}}]
        this_idxes[#this_idxes+1] = 1
        this_particles[#this_particles+1] = torch.squeeze(this,2)--this:resize(this:size(1), this:size(3), this:size(4)) -- (num_samples x windowsize x 8)
    end
    assert(#this_particles==num_particles)
    this_particles = torch.cat(this_particles,1)  -- concatenate along batch dimension

    -- make other_particles into Torch tensor if more than one particle. Otherwise {}
    if next(other_particles) then
        other_particles = torch.cat(other_particles,1)
        assert(this_particles:size()[1] == other_particles:size()[1])
        assert(other_particles:size()[2] == num_particles-1)
    end
    return this_particles, other_particles
end


--[[
    Each batch is a table of 5 things: {this, others, goos, mask, y}

        this: particle of interest, past timesteps
            (num_examples x windowsize/2 x 8)
            last dimension: [px, py, vx, vy, (onehot mass)]

        others: other particles, past timesteps
            (num_examples x (num_particles-1) x windowsize/2 x 8) or {}
            last dimension: [px, py, vx, vy, (onehot mass)]

        goos: goos, constant across time
            (num_examples x num_goos x 8) or empty tensor?
            last dimension: [left, top, right, bottom, (onehot gooStrength)]

        mask: mask for the number of particles
            tensor of length 10, 0s everywhere except at location (num_particles-1) + num_goos

        y: particle of interest, future timesteps
            (num_examples x windowsize/2 x 8)
            last dimension: [px, py, vx, vy, (onehot mass)]

    Note that num_samples = num_examples * num_particles

    Output: {this_x, context_x, y, minibatch_m}
        this_x: (num_samples_slice, windowsize/2 * object_dim)
        context_x: (num_samples_slice, max_other_objects, windowsize/2 * object_dim)
        y: (num_samples_slice, windowsize/2 * object_dim)
        minibatch_m: (max_other_objects)

        num_samples_slice is num_samples[start:finish], inclusive
--]]
function dataloader:next_config(current_config, start, finish)
    -- print('current_config', current_config)
    -- print('start', start, 'finish', finish)
    local minibatch_data = self.dataset[current_config]
    local minibatch_p = minibatch_data.particles  -- (num_examples x num_particles x windowsize x 8)
    local minibatch_g = minibatch_data.goos  -- (num_examples x num_goos x 8) or {}?
    local minibatch_m = minibatch_data.mask  -- 8

    local this_particles, other_particles = expand_for_each_particle(minibatch_p)
    local num_samples, windowsize = unpack(torch.totable(this_particles:size()))  -- num_samples is now multiplied by the number of particles
    local num_particles = minibatch_p:size(2)
    if num_samples ~= minibatch_p:size(1) * num_particles then
        print('num_samples', num_samples)
        print('minibatch_p:size(1) * num_particles', minibatch_p:size(1) * num_particles)
        print('minibatch_p:size()', minibatch_p:size(1))
        print('num_particles', num_particles)
        print('this_particles:size()', this_particles:size())
        print('other_particles:size()', other_particles:size())
    end

    assert(num_samples == minibatch_p:size(1) * num_particles)

    -- check if m_goos is empty
    -- if m_goos is empty, then {}, else it is (num_samples, num_goos, 8)
    local m_goos = {}
    local num_goos = 0  -- default
    if minibatch_g:dim() > 1 then
        for i=1,num_particles do m_goos[#m_goos+1] = minibatch_g end  -- make num_particles copies of minibatch_g
        m_goos = torch.cat(m_goos,1)
        num_goos = m_goos:size(2)
        m_goos = m_goos:reshape(m_goos:size(1), m_goos:size(2), 1, m_goos:size(3))  -- (num_samples, num_goos, 1, 8) -- take a look at this!
        local m_goos_window = {}
        for i=1,windowsize do m_goos_window[#m_goos_window+1] = m_goos end
        m_goos = torch.cat(m_goos_window, 3)
    end

    -- check if other_particles is empty
    local num_other_particles = 0
    if torch.type(other_particles) ~= 'table' then num_other_particles = other_particles:size(2) end

    -- get the number of steps that we need to pad to 0
    local num_to_pad = max_other_objects - (num_goos + num_other_particles)
    if num_goos + num_other_particles > 1 then assert(unpack(torch.find(minibatch_m,1)) == max_other_objects - num_to_pad) end  -- make sure we are padding the right amount

    -- create context
    local context
    if num_other_particles > 0 and num_goos > 0 then
        -- print('num_other_particles > 0 and num_goos > 0')
        context = torch.cat(other_particles, m_goos, 2)  -- (num_samples x (num_objects-1) x windowsize/2 x 8)
        if num_to_pad > 0 then
            local pad_p = torch.Tensor(num_samples, num_to_pad, windowsize, object_dim):fill(0)
            context = torch.cat(context, pad_p, 2)
        end
    else
        assert(num_to_pad > 0)
        local pad_p = torch.Tensor(num_samples, num_to_pad, windowsize, object_dim):fill(0)
        if num_other_particles > 0 then -- no goos
            -- print('num_other_particles > 0 and num_goos = 0')
            assert(torch.type(m_goos)=='table')
            assert(not next(m_goos))  -- the table had better be empty
            context = torch.cat(other_particles, pad_p, 2)
        elseif num_goos > 0 then -- no other objects
            -- print('num_other_particles = 0 and num_goos > 0')
            assert(torch.type(other_particles)=='table')
            assert(not next(other_particles))  -- the table had better be empty
            context = torch.cat(m_goos, pad_p, 2)
        else
            -- print('num_other_particles = 0 and num_goos = 0')
            assert(num_other_particles == 0 and num_goos == 0)
            assert(num_to_pad == max_other_objects)
            context = pad_p  -- context is just the pad then
        end
    end
    assert(context:dim() == 4 and context:size(1) == num_samples and
        context:size(2) == max_other_objects and context:size(3) == windowsize and
        context:size(4) == object_dim)

    -- split into x and y
    local num_past = math.floor(windowsize/2)
    local this_x = this_particles[{{},{1,num_past},{}}]  -- (num_samples x windowsize/2 x 8)
    local context_x = context[{{},{},{1,num_past},{}}]  -- (num_samples x max_other_objects x windowsize/2 x 8)
    local y = this_particles[{{},{num_past+1,-1},{}}]  -- (num_samples x windowsize/2 x 8)
    local context_future = context[{{},{},{num_past+1,-1},{}}]  -- (num_samples x max_other_objects x windowsize/2 x 8)

    -- assert num_samples are correct
    assert(this_x:size(1) == num_samples and context_x:size(1) == num_samples and y:size(1) == num_samples)
    -- assert number of axes of tensors are correct
    assert(this_x:size():size()==3 and context_x:size():size()==4 and y:size():size()==3)
    -- assert seq length is correct
    assert(this_x:size(2)==num_past and context_x:size(3)==num_past and y:size(2)==num_past)
    -- check padding
    assert(context_x:size(2)==max_other_objects)
    -- check data dimension
    assert(this_x:size(3) == object_dim and context_x:size(4) == object_dim and y:size(3) == object_dim)

    -- cuda
    if common_mp.cuda then
        this_x          = this_x:cuda()
        context_x       = context_x:cuda()
        minibatch_m     = minibatch_m:cuda()
        y               = y:cuda()
        context_future  = context_future:cuda()  -- TODO: note that you have to update this everywhere!
    end

    -- Reshape
    this_x          = this_x:reshape(num_samples, num_past*object_dim)
    context_x       = context_x:reshape(num_samples, max_other_objects, num_past*object_dim)
    y               = y:reshape(num_samples, num_past*object_dim)
    context_future  = context_future:reshape(num_samples, max_other_objects, (windowsize-num_past)*object_dim)

    assert(this_x:dim()==2 and context_x:dim()==3 and y:dim()==2)

    -- here only get the batch you need. There is a lot of redundant computation here
    this_x          = this_x[{{start,finish}}]
    context_x       = context_x[{{start,finish}}]
    y               = y[{{start,finish}}]
    context_future  = context_future[{{start,finish}}]

    collectgarbage()
    return {this_x, context_x, y, minibatch_m, current_config, start, finish, context_future}  -- here return start, finish, and configname too
end


-- works
function dataloader.slice_batch(table_of_data, start, finish)
    local sliced_table_of_data = {}
    for i=1,#table_of_data do
        sliced_table_of_data[i] = table_of_data[i][{{start,finish}}]
    end
    return sliced_table_of_data
end


function dataloader:count_examples(configs)
    local total_samples = 0
    local config_sizes = {}
    for i, config in pairs(configs) do
        local config_examples = self.dataset[config]
        local num_samples = config_examples.particles:size(1)*config_examples.particles:size(2)
        total_samples = total_samples + num_samples
        config_sizes[i] = num_samples -- each config has an id
    end
    assert(total_samples % self.batch_size == 0, 'Total Samples: '..total_samples.. ' batch size: '.. self.batch_size)
    local num_batches = total_samples/self.batch_size
    return total_samples, num_batches, config_sizes
end


function dataloader:compute_batches()
    local current_config = 1
    local current_batch_in_config = 0
    local batchlist = {}
    for i=1,self.num_batches do

        -- todo: update current_config and current_batch_in_config here
        local batch_info = self:get_batch_info(current_config, current_batch_in_config)
        current_config = unpack(subrange(batch_info, 4,4))
        current_batch_in_config = unpack(subrange(batch_info, 3,3))
        batchlist[#batchlist+1] = subrange(batch_info, 1,3)
    end
    assert(self.num_batches == #batchlist)
    return batchlist
end


function dataloader:get_batch_info(current_config, current_batch_in_config)

    -- assumption that a config contains more than one batch
    current_batch_in_config = current_batch_in_config + self.batch_size
    -- current batch is the range: [current_batch_in_config - self.batch_size + 1, current_batch_in_config]

    if current_batch_in_config > self.config_sizes[self.config_idxs[current_config]] then
        current_config = current_config + 1
        current_batch_in_config = self.batch_size -- reset current_batch_in_config
    end

    if current_config > self.num_configs then
        -- assert(self.current_batch == self.num_batches+1)
        current_config = 1
        -- self.current_batch = 0
        assert(current_batch_in_config == self.batch_size)
    end

    -- print('config: '.. self.configs[self.config_idxs[current_config]] ..
    --         ' capacity: '.. self.config_sizes[self.config_idxs[current_config]] ..
    --         ' current batch: ' .. '[' .. current_batch_in_config - self.batch_size + 1 ..
            -- ',' .. current_batch_in_config .. ']')
    return {self.specified_configs[self.config_idxs[current_config]],  -- config name
            current_batch_in_config - self.batch_size + 1,  -- start index in config
            current_batch_in_config, -- for next update
            current_config}  -- end index in config
end

function dataloader:next_batch()
    self.current_batch = self.current_batch + 1
    if self.current_batch > self.num_batches then self.current_batch = 1 end

    local config_name, start, finish = unpack(self.batchlist[self.batch_idxs[self.current_batch]])
    -- print('current batch: '..self.current_batch .. ' id: '.. self.batch_idxs[self.current_batch]..
    --         ' ' .. config_name .. ': [' .. start .. ':' .. finish ..']')
    local nextbatch = self:next_config(config_name, start, finish)
    return nextbatch
end

-- you should have an method that returns the batches for a whole config at once
-- that is just next_config, but with start and finish as the entire config.
-- then you'd break that up during training time.


-- orders the configs in topo order
-- there can be two equally valid topo sorts:
--      first: all particles, then all goos
--      second: diagonal
function topo_order(configs)
    table.sort(configs)
end


function contains_world(worldconfigtable)
    for _,v in pairs(worldconfigtable) do
        if #v >= #'worldm1' then
            local prefix = v:sub(1,#'worldm')
            local suffix = v:sub(#'worldm'+1)
            if (prefix == 'worldm') and (tonumber(suffix) ~= nil) then -- assert that it is a number
                return true
            end
        end
    end
    return false
end

-- worlds is a table of worlds
-- all_configs is a table of configs
function get_all_configs_for_worlds(worlds, all_configs)
    assert(is_subset(worlds, all_worlds))
    local world_configs = {}
    for i,config in pairs(all_configs) do
        for j,world in pairs(worlds) do
            if is_substring(world, config) then
                world_configs[#world_configs+1] = config
            end
        end
    end
    return world_configs
end


function get_all_specified_configs(worldconfigtable, all_configs)
    local all_specified_configs = {}
    for i, element in pairs(worldconfigtable) do
        if is_substring('np', element) then
            all_specified_configs[#all_specified_configs+1] = element
        else
            assert(#element == #'worldm1') -- check that it is a world
            all_specified_configs = merge_tables_by_value(all_specified_configs, get_all_configs_for_worlds({element}, all_configs))
        end
    end
    return all_specified_configs
end

return dataloader

-- -- d = dataloader.create('trainset','/om/user/mbchang/physics-data/dataset_files',false)
-- d = dataloader.create('trainset','haha', {'worldm1', 'worldm2_np=3_ng=3'}, 4, true, false)
-- print(d)

-- d.configs[#d.configs+1] = 'worldm2_np=6_ng=5'
-- d.configs[#d.configs+1] = 'worldm2_np=5_ng=3'
-- d.configs[#d.configs+1] = 'worldm3dfdf'
-- x = get_all_specified_configs({'worldm1_np=6_ng=5', 'worldm2'}, d.configs)


-- TODO: compute_batches is wrong; the start and finish are wrong?


-- -- for i=1,20 do
-- -- print(d:next_batch()) end
-- -- print(d:next_batch())
-- print(d.batchlist)
-- print('num_batches:', d.num_batches)
-- for i=1,d.num_batches do
--     d:next_batch()
-- end
-- -- print(d:count_examples(d.config_idxs))
-- -- d:compute_batches()

-- d:next_batch()



--[[
What I want to be able to is to have a dataloader, that takes in parameters:
    - shuffle
    - a table of configs (indexed by number, in order)

Then when I do next_batch, it will go through appropriately.


    You should have a function that returns all configs in sorted order
    as well as function that returns all configs of a certain world


]]
