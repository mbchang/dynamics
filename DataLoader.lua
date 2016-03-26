-- data loader for object model

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
require 'pl.stringx'
require 'pl.Set'
local T = require 'pl.tablex'
local PS = require 'priority_sampler'

local dataloader = {}
dataloader.__index = dataloader

local object_dim = 8
local max_other_objects = 10
local all_worlds = {'worldm1', 'worldm2', 'worldm3', 'worldm4'}  -- all_worlds[1] should correspond to worldm1
local world_range = {1,4}
local particle_range = {1,6}
local goo_range = {0,5}

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


function dataloader.create(dataset_name, specified_configs, dataset_folder, batch_size, shuffle, cuda, relative, num_past, winsize)
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
    assert(all_args_exist({dataset_name, dataset_folder, specified_configs,batch_size,shuffle,cuda},6))

    local self = {}
    setmetatable(self, dataloader)

    ---------------------------------- Givens ----------------------------------
    self.dataset_name = dataset_name  -- string
    self.dataset_folder = dataset_folder
    self.cuda = cuda
    self.current_batch = 0
    -- assert(self.num_batches == #self.batchlist)

    self.num_batches = 12 -- TODO hardcoded!

    self.priority_sampler = PS.create(self.num_batches)
    self.current_sampled_id = 0

    ---------------------------------- Shuffle ---------------------------------
    if shuffle then
        self.batch_idxs = torch.randperm(self.num_batches)
    else
        self.batch_idxs = torch.range(1,self.num_batches)
    end

    collectgarbage()
    return self
end

--
-- function dataloader:count_examples(configs)
--     local total_samples = 0
--     local config_sizes = {}
--     for i, config in pairs(configs) do
--         local config_examples = self.dataset[config]
--         local num_samples = config_examples.particles:size(1)*config_examples.particles:size(2)
--         total_samples = total_samples + num_samples
--         config_sizes[i] = num_samples -- each config has an id
--     end
--     assert(total_samples % self.batch_size == 0, 'Total Samples: '..total_samples.. ' batch size: '.. self.batch_size)
--     local num_batches = total_samples/self.batch_size
--     return total_samples, num_batches, config_sizes
-- end
-- --
-- --
-- function dataloader:compute_batches()
--     local current_config = 1
--     local current_batch_in_config = 0
--     local batchlist = {}
--     for i=1,self.num_batches do
--         local batch_info = self:get_batch_info(current_config, current_batch_in_config)
--         current_config = unpack(subrange(batch_info, 4,4))
--         current_batch_in_config = unpack(subrange(batch_info, 3,3))
--         batchlist[#batchlist+1] = subrange(batch_info, 1,3)
--     end
--     assert(self.num_batches == #batchlist)
--     return batchlist
-- end
-- --
-- --
-- function dataloader:get_batch_info(current_config, current_batch_in_config)
--     -- assumption that a config contains more than one batch
--     current_batch_in_config = current_batch_in_config + self.batch_size
--     -- current batch is the range: [current_batch_in_config - self.batch_size + 1, current_batch_in_config]
--
--     if current_batch_in_config > self.config_sizes[self.config_idxs[current_config]] then
--         current_config = current_config + 1
--         current_batch_in_config = self.batch_size -- reset current_batch_in_config
--     end
--
--     if current_config > self.num_configs then
--         current_config = 1
--         assert(current_batch_in_config == self.batch_size)
--     end
--
--     -- print('config: '.. self.configs[self.config_idxs[current_config]] ..
--     --         ' capacity: '.. self.config_sizes[self.config_idxs[current_config]] ..
--     --         ' current batch: ' .. '[' .. current_batch_in_config - self.batch_size + 1 ..
--     --         ',' .. current_batch_in_config .. ']')
--     return {self.specified_configs[self.config_idxs[current_config]],  -- config name
--             current_batch_in_config - self.batch_size + 1,  -- start index in config
--             current_batch_in_config, -- for next update
--             current_config}  -- end index in config
-- end


function dataloader:sample_priority_batch(pow)
    if self.priority_sampler.epc_num > 1 then  -- TODO turn this back to 1
        return self:load_batch_id(self.priority_sampler:sample(self.priority_sampler.epc_num/100))  -- sharpens in discrete steps  TODO this was hacky
        -- return self:sample_batch_id(self.priority_sampler:sample(pow))  -- sum turns it into a number
    else
        return self:sample_sequential_batch()  -- or sample_random_batch
    end
end

function dataloader:sample_random_batch()
    local id = math.random(self.num_batches)
    return self:load_batch_id(id)
end

function dataloader:sample_sequential_batch()
    self.current_batch = self.current_batch + 1
    if self.current_batch > self.num_batches then self.current_batch = 1 end
    return self:load_batch_id(self.batch_idxs[self.current_batch])
end
--
-- function dataloader:save_sequential_batches()
--     local savefolder = self.dataset_folder..'/'..'batches'..'/'..self.dataset_name
--     if not paths.dirp(savefolder) then paths.mkdir(savefolder) end
--     for i = 1,self.num_batches do
--         local batch = self:sample_batch_id(self.batch_idxs[i])
--         local batchname = savefolder..'/'..'batch'..i
--         torch.save(batchname, batch)
--         print('saved '..batchname)
--     end
-- end
--
-- function dataloader:sample_batch_id(id)
--     self.current_sampled_id = id
--     local config_name, start, finish = unpack(self.batchlist[id])
--     -- print('current batch: '..self.current_batch .. ' id: '.. self.batch_idxs[self.current_batch]..
--     --         ' ' .. config_name .. ': [' .. start .. ':' .. finish ..']')
--     local nextbatch = self:next_config(config_name, start, finish)
--     return nextbatch
-- end


function dataloader:load_batch_id(id)
    self.current_sampled_id = id
    -- local config_name, start, finish = unpack(self.batchlist[id])
    -- -- print('current batch: '..self.current_batch .. ' id: '.. self.batch_idxs[self.current_batch]..
    -- --         ' ' .. config_name .. ': [' .. start .. ':' .. finish ..']')
    -- local nextbatch = self:next_config(config_name, start, finish)

    local savefolder = self.dataset_folder..'/'..'batches'..'/'..self.dataset_name
    local batchname = savefolder..'/'..'batch'..id
    local nextbatch = torch.load(batchname)
    local this, context, y, mask, config, start, finish, context_future = unpack(nextbatch)

    -- convert to cuda or double
    this,context,y,context_future = unpack(map(convert_type,{this,context,y,context_future}))

    nextbatch = {this, context, y, mask, config, start, finish, context_future}
    collectgarbage()
    return nextbatch
end


-- this             (num_samples x num_past x 8)
-- context          (num_samples x max_other_objects x num_past x 8)
-- -- y                (num_samples x num_future x 8)
-- -- context_future   (num_samples x max_other_objects x num_future x 8)
-- function add_accel(this_x, context_x, y, context_future)
--     local this_x_accel = add_accel_each(this_x,true)
--     local context_x_accel = add_accel_each(context_x,false)
--     local y_accel = add_accel_each(y,true)
--     local context_future_accel = add_accel_each(context_future,false)
--
--     return {this_x_accel,context_x_accel,y_accel,context_future_accel}
-- end
--
-- function add_accel_each(obj,isthis)
--     local eps = 1e-10
--     local num_samples = obj:size(1)
--     if isthis then
--         assert(obj:dim() == 3)
--         local num_steps = obj:size(2)
--         local vel = obj[{{},{},{3,4}}]:clone()  -- num_samples, num_steps, 2
--         local accel = torch.zeros(num_samples,num_steps,2)
--
--         for step = 2,num_steps do
--             accel[{{},{step},{1}}] = torch.abs((vel[{{},{step},{1}}] - vel[{{},{step-1},{1}}])):gt(eps)
--             accel[{{},{step},{2}}] = torch.abs(vel[{{},{step},{2}}] - vel[{{},{step-1},{2}}]):gt(eps)
--         end
--
--         local new_obj = torch.zeros(num_samples,num_steps,obj:size(3)+2)
--         new_obj[{{},{},{3,4}}] = obj[{{},{},{3,4}}]
--         new_obj[{{},{},{5,6}}] = accel
--         new_obj[{{},{},{7,10}}] = obj[{{},{},{5,8}}]
--
--         return new_obj:clone()
--     else
--         assert(obj:dim() == 4)
--         local num_steps = obj:size(3)
--         local max_objects = obj:size(2)
--         local vel = obj[{{},{},{},{3,4}}]
--         local accel = torch.zeros(num_samples,max_objects,num_steps,2)
--
--         for step = 2,num_steps do
--             accel[{{},{},{step},{1}}] = torch.abs((vel[{{},{},{step},{1}}] - vel[{{},{},{step-1},{1}}])):gt(eps)
--             accel[{{},{},{step},{2}}] = torch.abs(vel[{{},{},{step},{2}}] - vel[{{},{},{step-1},{2}}]):gt(eps)
--         end
--
--         local new_obj = torch.zeros(num_samples,max_objects,num_steps,obj:size(4)+2)
--         new_obj[{{},{},{},{3,4}}] = obj[{{},{},{},{3,4}}]
--         new_obj[{{},{},{},{5,6}}] = accel
--         new_obj[{{},{},{},{7,10}}] = obj[{{},{},{},{5,8}}]
--         -- print('hi')
--         -- print(new_obj:size())
--         return new_obj:clone()
--     end
-- end
--
-- -- orders the configs in topo order
-- -- there can be two equally valid topo sorts:
-- --      first: all particles, then all goos
-- --      second: diagonal
-- function topo_order(configs)
--     table.sort(configs)
-- end


-- function contains_world(worldconfigtable)
--     for _,v in pairs(worldconfigtable) do
--         if #v >= #'worldm1' then
--             local prefix = v:sub(1,#'worldm')
--             local suffix = v:sub(#'worldm'+1)
--             if (prefix == 'worldm') and (tonumber(suffix) ~= nil) then -- assert that it is a number
--                 return true
--             end
--         end
--     end
--     return false
-- end
--
-- -- worlds is a table of worlds
-- -- all_configs is a table of configs
-- function get_all_configs_for_worlds(worlds, all_configs)
--     assert(is_subset(worlds, all_worlds))
--     local world_configs = {}
--     for i,config in pairs(all_configs) do
--         for j,world in pairs(worlds) do
--             if is_substring(world, config) then
--                 world_configs[#world_configs+1] = config
--             end
--         end
--     end
--     return world_configs
-- end
--
--
-- function get_all_specified_configs(worldconfigtable, all_configs)
--     local all_specified_configs = {}
--     for i, element in pairs(worldconfigtable) do
--         if is_substring('np', element) then
--             all_specified_configs[#all_specified_configs+1] = element
--         else
--             assert(#element == #'worldm1') -- check that it is a world
--             all_specified_configs = merge_tables_by_value(all_specified_configs, get_all_configs_for_worlds({element}, all_configs))
--         end
--     end
--     return all_specified_configs
-- end

-- [1-1-1] for worldm1, np 1 ng 1
-- implementation so far: either world or entire config
-- basically this implements slicing
function convert2config(config_abbrev)
    local wlow, whigh, nplow, nphigh, nglow, nghigh = string.match(config_abbrev, "%[(%d*):(%d*)-(%d*):(%d*)-(%d*):(%d*)%]")

    -- can't figure out how to do this with functional programming, because
    -- you can't pass nil arguments into function
    if not wlow or wlow == '' then wlow = -1 end
    if not nplow or nplow == '' then nplow = -1 end
    if not nglow or nglow == '' then nglow = -1 end
    if not whigh or whigh == '' then whigh = math.huge end
    if not nphigh or nphigh == '' then nphigh = math.huge end
    if not nghigh or nghigh == '' then nghigh = math.huge end

    wlow, nplow, nglow = math.max(world_range[1],wlow),
                         math.max(particle_range[1],nplow),
                         math.max(goo_range[1],nglow)  -- 0 because there can 0 goos
    whigh, nphigh, nghigh = math.min(world_range[2],whigh),
                            math.min(particle_range[2],nphigh),
                            math.min(goo_range[2],nghigh)  -- 0 because there can 0 goos

    local all_configs = {}
    for w in range(wlow, whigh) do
        for np in range(nplow, nphigh) do
            for ng in range(nglow, nghigh) do
                all_configs[#all_configs+1] = all_worlds[w] .. '_np=' .. np .. '_ng=' .. ng
            end
        end
    end
    return all_configs
end

-- "[4--],[1-2-3],[3--],[2-1-5]"
-- notice that is surrounded by brackets
function dataloader.convert2allconfigs(config_abbrev_table_string)
    assert(stringx.lfind(config_abbrev_table_string, ' ') == nil)
    local x = stringx.split(config_abbrev_table_string,',')  -- get rid of brackets; x is a table

    -- you want to merge into a set
    local y = map(convert2config,x)
    local z = {}
    for _, table in pairs(y) do
        z = merge_tables_by_value(z,table)
    end
    return z
end

return dataloader

--
-- d = dataloader.create('trainset', {}, '/om/user/mbchang/physics-data/dataset_files_subsampled', 100, false, false)
-- d = dataloader.create('trainset', {'worldm1', 'worldm2_np=3_ng=3'}, 'haha', 4, true, false)
-- print(d.specified_configs)


-- d = dataloader.create('trainset', convert2allconfigs("[4:-:-:],[1-2-3]"), 'haha', 4, true, false)
-- print(d.specified_configs)



-- local cur = {'[:-1:1-:]','[:-2:2-:]','[:-3:3-:]','[:-4:4-:]','[:-5:5-:]','[:-6:6-:]'}
-- for _,c in pairs(cur) do
--     print(c)
--     print(convert2allconfigs(c))
--     print('################################################################')
-- end



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
