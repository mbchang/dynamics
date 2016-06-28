-- data loader for object model

require 'torch'
require 'math'
require 'image'
require 'lfs'
require 'sys'
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

local datasampler = {}
datasampler.__index = datasampler

-- here we want a method for splitting into past and future

-- when you create the data sampler, you should already initialize the table ()

function datasampler.create(dataset_name, dataset_folder, shuffle, cuda)
    --[[
        I actually only need dataset_name, dataset_folder, shufffle, cuda. Do I need a priority_sampler?

        Input
            dataset_name: file containing data, like 'trainset'
            dataset_folder: folder containing the .h5 files
            shuffle: boolean
    --]]
    assert(all_args_exist({dataset_name, dataset_folder,shuffle,cuda},4))  -- TODO: This needs to be refactored eventually

    local self = {}
    setmetatable(self, datasampler)

    ---------------------------------- Givens ----------------------------------
    self.dataset_name = dataset_name  -- string
    self.dataset_folder = dataset_folder
    self.cuda = cuda
    self.current_batch = 0

    --hard coded
    local args = {winsize = 10,
                  num_past = 2,
                  num_future = 1,
                  maxwinsize = 60,
                  sim = true}  -- TODO: consider how you want to realize this (perhaps you pass this in from eval?)

    self.winsize = args.winsize
    self.num_past = args.num_past
    self.num_future = args.num_future
    self.sim = args.sim
    assert(self.num_past + self.num_future <= self.winsize)
    assert(self.winsize < args.maxwinsize)  -- not sure if this is going to come from config or not

    -- here find out how many batches (for now, we won't do any dynamic re-distributing)
    -- local savefolder = self.dataset_folder..'/'..'batches'..'/'..self.dataset_name
    self.savefolder = 'debug_tower'..'/'..self.dataset_name  -- TODO: replace lat
    self.savefolder = string.sub(self.savefolder, 1,-4)
    -- print(self.savefolder)
    -- assert(false)
    self.num_batches = tonumber(sys.execute("ls -1 " .. self.savefolder .. "/ | wc -l"))
    print(self.num_batches)

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

function datasampler:split_time(batch)
    local focus, context = unpack(batch)
    assert(focus:size(2) >= self.winsize and context:size(3) >= self.winsize)
    local focus_past = focus[{{},{1, self.num_past}}]
    local context_past = context[{{},{}, {1, self.num_past}}]
    local focus_future, context_future
    if self.sim then
        focus_future = focus[{{},{self.num_past+1, -1}}]
        context_future = context[{{},{},{self.num_past+1, -1}}]
    else
        focus_future = focus[{{},{self.num_past+1, self.num_past+self.num_future}}]
        context_future = context[{{},{},{self.num_past+1, self.num_past+self.num_future}}]
    end

    return {focus_past, context_past, focus_future, context_future}
end

function datasampler:sample_priority_batch(pow)
    -- return self:sample_random_batch()  -- or sample_random_batch
    --
    if self.priority_sampler.epc_num > 1 then  -- TODO turn this back to 1
        -- return self:load_batch_id(self.priority_sampler:sample(self.priority_sampler.epc_num/100))  -- sharpens in discrete steps  TODO this was hacky
        return self:load_batch_id(self.priority_sampler:sample(pow))  -- sum turns it into a number
    else
        return self:sample_sequential_batch()  -- or sample_random_batch
    end
end

function datasampler:sample_random_batch()
    local id = math.random(self.num_batches)
    return self:load_batch_id(id)
end

function datasampler:sample_sequential_batch()
    self.current_batch = self.current_batch + 1
    if self.current_batch > self.num_batches then self.current_batch = 1 end
    return self:load_batch_id(self.batch_idxs[self.current_batch])
end

function datasampler:load_batch_id(id)
    self.current_sampled_id = id
    -- local config_name, start, finish = unpack(self.batchlist[id])
    -- -- print('current batch: '..self.current_batch .. ' id: '.. self.batch_idxs[self.current_batch]..
    -- --         ' ' .. config_name .. ': [' .. start .. ':' .. finish ..']')
    -- local savefolder = self.dataset_folder..'/'..'batches'..'/'..self.dataset_name

    local batchname = self.savefolder..'/'..'batch'..id
    -- if not paths.filep(batchname) then batchname = batchname..'_hard' end -- this is a hard example. TODO: maybe put in a binary value as input
    local nextbatch = torch.load(batchname)

    nextbatch = self:split_time(nextbatch)

    local this, context, y, context_future, mask = unpack(nextbatch)

    -- NOTE hardcoded!
    mask = torch.zeros(10)
    mask[{{context_future:size(2)}}] = 1 -- I'm assuming that mask is for the number of context, but you can redefine this

    -- convert to cuda or double
    this,context,y,context_future = unpack(map(convert_type,{this,context,y,context_future},self.cuda))

    nextbatch = {this, context, y, context_future, mask}
    collectgarbage()
    return nextbatch
end



return datasampler
