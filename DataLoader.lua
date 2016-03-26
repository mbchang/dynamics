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
    assert(all_args_exist({dataset_name, dataset_folder,shuffle,cuda},4))

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



return dataloader
