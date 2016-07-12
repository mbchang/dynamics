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
local pls = require 'pl.stringx'
require 'pl.Set'
local T = require 'pl.tablex'
local PS = require 'priority_sampler'
local data_process = require 'data_process'

local datasampler = {}
datasampler.__index = datasampler

-- here we want a method for splitting into past and future

-- when you create the data sampler, you should already initialize the table ()

-- function datasampler.create(dataset_name, dataset_folder, shuffle, cuda)
function datasampler.create(dataset_name, args)
    --[[
        I actually only need dataset_name, dataset_folder, shufffle, cuda. Do I need a priority_sampler?

        Input
            dataset_name: file containing data, like 'trainset'
            dataset_folder: folder containing the .h5 files
            shuffle: boolean
    --]]

    local self = {}
    setmetatable(self, datasampler)
    self.current_batch = 0
    self.dataset_folder=args.dataset_folder
    self.dataset_name=dataset_name
    self.maxwinsize=args.maxwinsize
    self.winsize=args.winsize
    self.num_past=args.num_past
    self.num_future=args.num_future
    self.relative=args.relative
    self.sim=args.sim
    self.cuda=args.cuda
    assert(self.num_past + self.num_future <= self.winsize)
    assert(self.winsize < args.maxwinsize)  -- not sure if this is going to come from config or not

    -- here find out how many batches (for now, we won't do any dynamic re-distributing)
    self.savefolder = self.dataset_folder..'/'..'batches'..'/'..self.dataset_name
    print('savefolder', self.savefolder)
    self.num_batches = tonumber(sys.execute("ls -1 " .. self.savefolder .. "/ | wc -l"))
    print(self.dataset_name..': '..self.dataset_folder..' number of batches: '..self.num_batches)

    self.priority_sampler = PS.create(self.num_batches)
    self.current_sampled_id = 0
    self.batch_idxs = torch.range(1,self.num_batches)
    self.current_dataset = 1

    -- debug
    self.has_reported = false

    collectgarbage()
    return self
end

function datasampler:split_time(batch)
    local focus, context = unpack(batch)
    assert(focus:size(2) >= self.winsize and context:size(3) >= self.winsize)  -- IS THIS WHAT WE WANT?
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

function datasampler:relative_batch(batch, rta)
    local this_past, context_past, this_future, context_future, mask = unpack(batch)

    -- TODO: use config args for this!
    this_future = data_process.relative_pair(this_past, this_future, rta)

    return {this_past, context_past, this_future, context_future, mask}
end

function datasampler:sample_priority_batch(pow)
    -- return self:sample_random_batch()  -- or sample_random_batch

    local batch
    --
    if self.priority_sampler.table_is_full then
        -- return self:load_batch_id(self.priority_sampler:sample(self.priority_sampler.epc_num/100))  -- sharpens in discrete steps  TODO this was hacky
        batch = self:load_batch_id(self.priority_sampler:sample(pow))  -- sum turns it into a number
    else
        batch = self:sample_sequential_batch()  -- or sample_random_batch
    end

    if self.priority_sampler.table_is_full and not(self.has_reported) then
        print(self.dataset_folder..' has seen all batches')  -- DEBUG
        self.has_reported = true
    end

    return batch
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

    local batchname = self.savefolder..'/'..'batch'..id
    local nextbatch = torch.load(batchname)

    nextbatch = self:split_time(nextbatch)
    if self.relative then nextbatch = self:relative_batch(nextbatch, false) end

    local this, context, y, context_future, mask = unpack(nextbatch)

    mask = torch.zeros(10)
    mask[{{context_future:size(2)}}] = 1 -- I'm assuming that mask is for the number of context, but you can redefine this

    -- convert to cuda or double
    this,context,y,context_future, mask = unpack(map(convert_type,{this,context,y,context_future, mask},self.cuda))

    nextbatch = {this, context, y, context_future, mask}
    collectgarbage()
    return nextbatch
end

function datasampler:get_hardest_batch()
    return self.priority_sampler:get_hardest_batch()
end

function datasampler:update_batch_weight(weight)
    self.priority_sampler:update_batch_weight(self.current_sampled_id, weight)
end

return datasampler
