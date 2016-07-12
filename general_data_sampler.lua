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
local D = require 'data_sampler'
local plseq = require 'pl.seq'

--
-- b = "{3,4}"
-- b = "{'balls_n20_t60_ex100','tower_n6_t60_ex20','balls_n1_t60_ex10'}"
-- loadstring("b="..string.gsub(b,'\"',''))()
-- print(b)

local general_datasampler = {}
general_datasampler.__index = general_datasampler

-- here we want a method for splitting into past and future

-- when you create the data sampler, you should already initialize the table ()
-- b = "{'balls_n20_t60_ex100','tower_n6_t60_ex20','balls_n1_t60_ex10'}"
-- local c = loadstring("return "..string.gsub(b,'\"',''))()
-- print('c',c)
-- a = assert(loadstring("return "..string.gsub(b,'\"',''))())
-- local context = {}
-- setfenv(a, context)
-- print('a',a)
-- assert(false)
-- print(b)

-- function get_datasets(datasets_string)
--     local datasets_string_eval = assert(loadstring("return "..string.gsub(datasets_string,'\"',''))())
--     return datasets_string_eval
-- end
-- print(get_datasets(b))
-- assert(false)

-- function general_datasampler.create(dataset_name, dataset_folder, shuffle, cuda)
function general_datasampler.create(dataset_name, args)
    --[[
        I actually only need dataset_name, dataset_folder, shufffle, cuda. Do I need a priority_sampler?

        Input
            dataset_name: file containing data, like 'trainset'
            dataset_folder: folder containing the .h5 files
            shuffle: boolean
    --]]

    local self = {}
    setmetatable(self, general_datasampler)
    self.current_batch = 0
    self.dataset_folders = assert(loadstring("return "..string.gsub(args.dataset_names,'\"',''))())
    self.dataset_name = dataset_name
    self.maxwinsize=args.maxwinsize
    self.winsize=args.winsize
    self.num_past=args.num_past
    self.num_future=args.num_future
    self.relative=args.relative
    self.sim=args.sim
    self.cuda=args.cuda
    assert(self.num_past + self.num_future <= self.winsize)
    assert(self.winsize < args.maxwinsize)  -- not sure if this is going to come from config or not

    self.datasamplers = {}
    for i, dataset_folder in pairs(self.dataset_folders) do
        args.dataset_folder='mj_data/'..dataset_folder -- NOTE HARDCODED!
        self.datasamplers[i] = D.create(dataset_name, args)
    end

    -- here find out how many batches (for now, we won't do any dynamic re-distributing)
    -- self.savefolder = self.dataset_folder..'/'..'batches'..'/'..self.dataset_name
    self.num_batches = plseq.reduce(function(x,y) return x + y end,
                            plseq.map(function(x) return x.num_batches end,
                                self.datasamplers))
    print(self.dataset_name,'num_batches', self.num_batches)
    -- self.num_batches = tonumber(sys.execute("ls -1 " .. self.savefolder .. "/ | wc -l"))

    -- self.priority_sampler = PS.create(self.num_batches)
    self.current_sampled_id = nil
    self.current_dataset = 1
    -- -- self.batch_idxs = torch.range(1,self.num_batches)
    -- self.datasampler_current_sampled_ids = plseq.map(function(x) return 1 end,
    --                                         self.datasamplers)

    -- TODO: ou can have an isfull field that is set if all of the consituent datasamplers are full
    self.has_seen_all_batches = false
    self.has_reported = false

    collectgarbage()
    return self
end

function general_datasampler:sample_priority_batch(pow)
    self.current_dataset = math.random(#self.datasamplers)
    local batch = self.datasamplers[self.current_dataset]:sample_priority_batch(pow)
    self.current_sampled_id = self.datasamplers[self.current_dataset].current_sampled_id
    -- print(self.current_dataset..'.'..self.datasamplers[self.current_dataset].current_sampled_id)
    --
    --
    if plseq.reduce('and', plseq.map(function(x) return x.has_reported end,
            self.datasamplers)) and not(self.has_reported) then
        self.has_seen_all_batches = true
        self.has_reported = true
        print('Seen all batches')
    end
    -- local batch = self:sample_sequential_batch()
    return batch
end

-- returns {loss, idx}
function general_datasampler:get_hardest_batch()
    -- TODO!  need the current dataset and current sampled id
    local hardest_batch = self.datasamplers[self.current_dataset]:get_hardest_batch()
    return {hardest_batch[1], hardest_batch[2], self.current_dataset}
end

function general_datasampler:update_batch_weight(weight)
    -- TODO!
    self.datasamplers[self.current_dataset]:update_batch_weight(weight)
end

function general_datasampler:sample_sequential_batch()
    local datasampler = self.datasamplers[self.current_dataset]
    self.current_sampled_id = self.datasamplers[self.current_dataset].current_sampled_id
    local batch = datasampler:sample_sequential_batch()
    if datasampler.current_batch == datasampler.num_batches then
        self.current_dataset = self.current_dataset % #self.datasamplers + 1
    end
    return batch
end


return general_datasampler


-- dataset_names = "{'balls_n3_t60_ex20','balls_n6_t60_ex20','balls_n5_t60_ex20'}"
-- -- dataset_names = "{'balls_n20_t60_ex100','balls_n1_t60_ex10'}"
-- -- dataset_names = "{'balls_n20_t60_ex100'}"
--
-- local data_loader_args = {
--                         dataset_names = dataset_names,
--                         dataset_folder='mj_data'..'/',--..dataset_names,  -- NOTE THESE ARE THE DATASET NAMES!
--                         maxwinsize=60,
--                         winsize=10, -- not sure if this should be in mp
--                         num_past=2,
--                         num_future=1,
--                         relative=true, -- TODO: this should be in the saved args!
--                         sim=false,
--                         cuda=false
--                         }
--
-- gd = general_datasampler.create('trainset', data_loader_args)
-- for i = 1, 100 do
--     gd:sample_priority_batch(1)
-- end
