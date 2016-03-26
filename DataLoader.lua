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

function dataloader.create(dataset_name, dataset_folder, shuffle, cuda)
    --[[
        I actually only need dataset_name, dataset_folder, shufffle, cuda. Do I need a priority_sampler?

        Input
            dataset_name: file containing data, like 'trainset'
            dataset_folder: folder containing the .h5 files
            shuffle: boolean
    --]]
    assert(all_args_exist({dataset_name, dataset_folder, specified_configs,batch_size,shuffle,cuda},6))

    local self = {}
    setmetatable(self, dataloader)

    ---------------------------------- Givens ----------------------------------
    self.dataset_name = dataset_name  -- string
    self.dataset_folder = dataset_folder
    self.cuda = cuda
    self.current_batch = 0

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

function dataloader:load_batch_id(id)
    self.current_sampled_id = id
    -- local config_name, start, finish = unpack(self.batchlist[id])
    -- -- print('current batch: '..self.current_batch .. ' id: '.. self.batch_idxs[self.current_batch]..
    -- --         ' ' .. config_name .. ': [' .. start .. ':' .. finish ..']')
    local savefolder = self.dataset_folder..'/'..'batches'..'/'..self.dataset_name
    local batchname = savefolder..'/'..'batch'..id
    local nextbatch = torch.load(batchname)
    local this, context, y, mask, config, start, finish, context_future = unpack(nextbatch)

    -- convert to cuda or double
    this,context,y,context_future = unpack(map(convert_type,{this,context,y,context_future},self.cuda))

    nextbatch = {this, context, y, mask, config, start, finish, context_future}
    collectgarbage()
    return nextbatch
end

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
