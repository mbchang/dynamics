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
    self.dataset_folder=args.dataset_folder
    self.dataset_name=dataset_name

    if not(string.find(self.dataset_folder, 'tower') == nil) then
        self.maxwinsize = config_args.maxwinsize_long
    else
        self.maxwinsize = config_args.maxwinsize
    end

    -- self.maxwinsize=args.maxwinsize
    self.winsize=args.winsize
    self.num_past=args.num_past
    self.num_future=args.num_future
    self.relative=args.relative
    self.shuffle=args.shuffle
    self.subdivide = args.subdivide
    self.sim=args.sim
    self.cuda=args.cuda
    assert(self.num_past + self.num_future <= self.winsize)
    assert(self.winsize < args.maxwinsize)  -- not sure if this is going to come from config or not
    if self.subdivide then assert(self.shuffle) end  -- you have to shuffle if you subdivide

    self.savefolder = self.dataset_folder..'/'..'batches'..'/'..self.dataset_name
    print('savefolder', self.savefolder)
    self.num_batches = tonumber(sys.execute("ls -1 " .. self.savefolder .. "/ | wc -l"))
    self.num_subbatches_per_batch = math.floor(self.maxwinsize/self.winsize)  -- NOTE: assume that all batches contain self.maxwinsize timesteps!
    self.num_subbatches = self.num_batches*self.num_subbatches_per_batch
    print(self.dataset_name..': '..self.dataset_folder..
            ' number of batches: '..self.num_batches..
            ' number of subbatches: '..self.num_subbatches)

    if self.subdivide then
        self.total_batches = self.num_subbatches
    else
        self.total_batches =self.num_batches
    end

    if self.shuffle then
        self.batch_idxs = torch.randperm(self.total_batches)
    else
        self.batch_idxs = torch.range(1,self.total_batches)
    end

    self.priority_sampler = PS.create(self.total_batches)

    self.current_sampled_id = 0
    self.current_batch = 0  -- may be deprecated
    self.current_subbatch = 0
    self.current_dataset = 1

    self.has_reported = false

    collectgarbage()
    return self
end

function datasampler:split_time(batch, offset)
    local offset = offset or 1
    local focus, context = unpack(batch)
    assert(focus:size(2) >= self.winsize and context:size(3) >= self.winsize)  -- IS THIS WHAT WE WANT?
    assert((offset-1)+self.num_past+self.num_future <= self.maxwinsize)

    local focus_past = focus[{{},{offset, (offset-1)+self.num_past}}]
    local context_past = context[{{},{}, {offset, (offset-1)+self.num_past}}]
    local focus_future, context_future
    if self.sim then
        focus_future = focus[{{},{(offset-1)+self.num_past+1, -1}}]
        context_future = context[{{},{},{(offset-1)+self.num_past+1, -1}}]
    else
        focus_future = focus[{{},{(offset-1)+self.num_past+1, (offset-1)+self.num_past+self.num_future}}]
        context_future = context[{{},{},{(offset-1)+self.num_past+1, (offset-1)+self.num_past+self.num_future}}]
    end

    return {focus_past, context_past, focus_future, context_future}
end

function datasampler:relative_batch(batch, rta)
    local this_past, context_past, this_future, context_future, mask = unpack(batch)
    this_future = data_process.relative_pair(this_past, this_future, rta)
    return {this_past, context_past, this_future, context_future, mask}
end

function datasampler:sample_random_batch(pow)
    self.current_batch = math.random(self.total_batches)
    local batch = self:load_batch_id(self.batch_idxs[self.current_batch])
    return batch
end

function datasampler:sample_priority_batch(pow)
    local batch
    if self.priority_sampler.table_is_full then
        batch = self:load_batch_id(self.priority_sampler:sample(pow))
    else
        batch = self:sample_sequential_batch()
    end

    if self.priority_sampler.table_is_full and not(self.has_reported) then
        print(self.dataset_folder..' has seen all batches')  -- DEBUG
        self.has_reported = true
    end

    return batch
end

-- note that this could still be random, but we will sample sequentially without replacement
function datasampler:sample_sequential_batch()
    self.current_batch = (self.current_batch % self.total_batches) + 1
    local batch = self:load_batch_id(self.batch_idxs[self.current_batch])
    return batch
end

function datasampler:load_batch_id(id)
    local batch
    if self.subdivide then
        batch = self:load_subbatch_id(id)
    else
        batch = self:load_batch_id_first_offset(id)
    end
    return batch
end

function datasampler:load_batch_id_first_offset(id)
    self.current_sampled_id = id

    local batchname = self.savefolder..'/'..'batch'..id
    local nextbatch = torch.load(batchname)   -- focus: (bsize, maxwinsize, obj_dim)

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

function datasampler:load_subbatch_id(id)
    self.current_sampled_id = id  -- note! I don't even need to change this! This now indexes the subbatches!
    local batch_id = math.floor((self.current_sampled_id-1) / self.num_subbatches_per_batch) + 1
    local offset = (self.current_sampled_id-1) - (self.num_subbatches_per_batch*(batch_id-1)) + 1
    -- print('batch_id', batch_id, 'offset', offset, 'csi,', self.current_sampled_id, 'name', self.dataset_folder, 'set', self.dataset_name)

    local batchname = self.savefolder..'/'..'batch'..batch_id
    local nextbatch = torch.load(batchname)   -- focus: (bsize, maxwinsize, obj_dim)

    nextbatch = self:split_time(nextbatch, offset)
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

-- either: focus (num_samples*num_obj, num_steps, obj_dim)
        --> (num_samples*num_obj*(num_steps/window_size), window_size, obj_dim)
-- or: context (num_samples*num_obj, num_obj-1, num_steps, obj_dim)
        --> (num_samples*num_obj*(num_steps/window_size), num_obj-1, window_size, obj_dim)
-- windowsize: num_past + num_future
function datasampler:subdivide_time(data)
    local num_ex = data:size(1)

    -- define window_size here
    local dim_to_split = data:dim()-1
    local splitted = data:clone():split(window_size, dim_to_split)
    local joined = torch.cat(splitted,1)
    assert(joined:size(1)==num_ex*window_size and joined:size(3)==window_size)
    return joined
end


function datasampler:get_hardest_batch()
    return self.priority_sampler:get_hardest_batch()
end

function datasampler:update_batch_weight(weight)
    -- print('self.current_sampled_id', self.current_sampled_id)
    self.priority_sampler:update_batch_weight(self.current_sampled_id, weight)
end

return datasampler
