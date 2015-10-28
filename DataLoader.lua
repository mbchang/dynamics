--[[
Looks in the data folder, which contains a bunch of images and loops through them
to make a 4d video
--]]

require 'metaparams'
require 'torch'
require 'math'
require 'image'
require 'lfs'

if common_mp.cuda then
    require 'cutorch'
    require 'cunn'
end

if common_mp.cudnn then
    require 'cudnn'
end

local DataLoader = {}
DataLoader.__index = DataLoader

function split_table(table, num_chunks)
    --[[
        input
            :type table: table
            :param table: table of elements
            
            :type num_chunks: int
            :param num_chunks: number of chunks you want to split the table into
        output
            :type: table of subtables
            :value: the number of subtables is num_chunks, each of size math.floor(#table/num_chunks)
    --]]
    local n = #table
    local chunk_size = math.floor(n/num_chunks)
    local splitted_table = {}
    local current_chunk = {}
    for i = 1, n do
        current_chunk[#current_chunk+1] = table[i]
        if i % chunk_size == 0 then
            splitted_table[#splitted_table+1] = current_chunk
            current_chunk = {}
        end
    end
    collectgarbage()
    return splitted_table
end


function find_all_sequences(folders_list, parent_folder_path, seq_length)
    local data_list = {}
    for f = 1, #folders_list do
        local data_path = parent_folder_path .. '/' .. folders_list[f]

        -- get number of images in this folder
        local num_images_f = io.popen('ls "' .. data_path .. '" | wc -l')
        local num_images = nil
        for x in num_images_f:lines() do num_images = x end
        local num_examples = math.floor(num_images/(seq_length))
        num_images = num_examples*seq_length

        -- cycle through images
        local p = io.popen('find "' .. data_path .. '" -type f -name "*.png"')  -- Note: this is not in order!
        local j = 0
        local ex_string = {}
        for img_name in p:lines() do
            j = j + 1
            ex_string[#ex_string+1] = data_path .. '/' .. j .. '.png'  -- force the images to be in order
            if j % seq_length == 0 then
                data_list[#data_list+1] = ex_string
                ex_string = {}
            end
        end
    end
    collectgarbage()
    return data_list
end

-- function isfile(filename)
--     local file = io.open(filename, 'r')
--     return file != nil
-- end

function DataLoader.create(dataset, seq_length, batch_size, shuffle)
    --[[
        input
            :type dataset: str
            :param dataset: name of folder containing training examples

            :type seq_length: int
            :param seq_length: length of sequence
    --]]
    local self = {}
    setmetatable(self, DataLoader)

    ---------------------------------- Initialize stuff ----------------------------------
    self.current_batch = 1
    self.dataset_name = dataset  -- string
    self.seq_length = seq_length
    self.batch_size = batch_size
    -- self.data_folder_path = "/home/mbchang/datasets/bounce/" .. dataset
    self.data_folder_path = "/Users/MichaelChang/Documents/SuperUROPlink/Code/NOF/datagen/bounce/" .. dataset
    self.data_folder = {}  -- keys are example ids, values are the folder name
    for folder in lfs.dir(self.data_folder_path) do
        if (#folder ==13) then -- this has time information
            self.data_folder[#self.data_folder+1] = folder
        end
    end

    --------------------------------- Find all sequences ---------------------------------
    -- Note that if the number of images in a certain folder < seq_length, that folder is skipped
    self.data_list = find_all_sequences(self.data_folder, self.data_folder_path, self.seq_length)
    self.num_examples = #self.data_list
    self.nbatches = math.floor(self.num_examples / self.batch_size)

    --------------------------------- Split into batches ---------------------------------
    self.batches = split_table(self.data_list, self.nbatches)
    if shuffle then 
        self.batch_idxs = torch.randperm(self.nbatches)
    else
        self.batch_idxs = torch.range(1,self.nbatches)
    end

    --------------------------- Save list of batches into file ---------------------------
    local batches_file = common_mp.results_folder .. '/examples_for_'..self.dataset_name..'_'..'seq_len='..self.seq_length .. '_' ..'batchsize='..self.batch_size
    if io.open(batches_file, 'r') == nil then 
        torch.save(batches_file, self.batches) 
        print('Saved batches file.')
    end

    collectgarbage()
    return self
end


function DataLoader:next_batch(mp)
    ----------------------------------- Load Batchfile -----------------------------------
    -- print(self.batch_idxs[self.current_batch])
    local batch_files = self.batches[self.batch_idxs[self.current_batch]]
    self.current_batch = self.current_batch + 1
    if self.current_batch > self.nbatches then self.current_batch = 1 end

    --------------------------------- Populate With Data ---------------------------------
    local minibatch = torch.Tensor(torch.LongStorage{#batch_files, self.seq_length, mp.color_channels, mp.frame_height, mp.frame_width})
    for i = 1, #batch_files do
        for t = 1, self.seq_length do 
            img = image.load(batch_files[i][t])
            img = img/255 -- normalize
            minibatch[{i,t}] = img
        end
    end 

    -- Here define training example pairs
    local x  = minibatch
    local y = minibatch

    if common_mp.cuda then
        x = x:cuda()
        y = y:cuda()
    end
    return x, y
end

return DataLoader

