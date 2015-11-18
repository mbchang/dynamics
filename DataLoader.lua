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

if common_mp.cuda then
    require 'cutorch'
    require 'cunn'
end

if common_mp.cudnn then
    require 'cudnn'
end

local dataloader = {}
dataloader.__index = dataloader

-- TODO: normalize pixel coordinates!


--[[ Loads the dataset as a table of configurations

    Input
        .h5 file data
        The data for each configuration is spread out across 3 keys in the .h5 file
        Let <config> be the configuration name
            "<config>particles": (num_samples, num_particles, windowsize, [px, py, vx, vy, mass])
            "<config>goos": (num_samples, num_goos, [left, top, right, bottom, gooStrength])
            "<config>mask": binary vector of length 5, trailing zeros are padding when #particles < 6

    Output
    {
        configuration:
            {
              particles : DoubleTensor - size: (num_samples x num_particles x windowsize x 5)
              goos : DoubleTensor - size: (num_samples x num_goos x 5)
              mask : DoubleTensor - size: 5 
            }
    }

    The last dimension of particles is 5 because: [px, py, vx, vy, mass]
    The last dimension of goos is 5 because: [left, top, right, bottom, gooStrength]
    The mask is dimension 5 because: our data has at most 6 particles -- ]]
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


function dataloader.create(dataset_name, dataset_folder, shuffle)
    --[[
        Input
            dataset_name: file containing data, like 'trainset'
            dataset_folder: folder containing the .h5 files
            shuffle: boolean
    --]]
    local self = {}
    setmetatable(self, dataloader)

    self.dataset_name = dataset_name  -- string
    self.dataset = load_data(dataset_name..'.h5', dataset_folder)  -- table of all the data
    self.configs = get_keys(self.dataset)  -- table of all keys
    self.nbatches = #self.configs
    if shuffle then 
        self.batch_idxs = torch.randperm(self.nbatches)
    else
        self.batch_idxs = torch.range(1,self.nbatches)
    end
    self.current_batch = 0

    collectgarbage()
    return self
end


--[[ Expands the number of examples per batch to have an example per particle
    Input: batch_particles: (num_samples x num_particles x windowsize x 5)
    Output: 
        {
            
        }
--]]
function expand_for_each_particle(batch_particles)
    local num_samples, num_particles, windowsize = unpack(torch.totable(batch_particles:size()))
    local this_particles = {}
    local other_particles = {}
    for i=1,num_particles do
        local this = torch.squeeze(batch_particles[{{},{i},{},{}}])  -- (num_samples x windowsize x 5)
        local other
        if i == 1 then
            other = batch_particles[{{},{i+1,-1},{},{}}]
        elseif i == num_particles then
            other = batch_particles[{{},{1,i-1},{},{}}]
        else
            other = torch.cat(batch_particles[{{},{1,i-1},{},{}}], 
                        batch_particles[{{},{i+1,-1},{},{}}], 2)  -- leave this particle out (num_samples x (num_particles-1) x windowsize x 5)
        end
        assert(this:size()[1] == other:size()[1])
        this_particles[#this_particles+1] = this
        other_particles[#other_particles+1] = other
    end

    this_particles = torch.cat(this_particles,1)  -- concatenate along batch dimension
    other_particles = torch.cat(other_particles,1)
    assert(this_particles:size()[1] == other_particles:size()[1])
    assert(other_particles:size()[2] == num_particles-1)
    return this_particles, other_particles
end


--[[
    Each batch is a table of 5 things: {this, others, goos, mask, y}

        this: particle of interest, past timesteps
            (num_samples x windowsize/2 x 5)
            last dimension: [px, py, vx, vy, mass]

        others: other particles, past timesteps
            (num_samples x (num_particles-1) x windowsize/2 x 5)
            last dimension: [px, py, vx, vy, mass]

        goos: goos, constant across time
            (num_samples x num_goos x 5)
            last dimension: [left, top, right, bottom, gooStrength]

        mask: mask for the number of particles
            tensor of length 5, 0s padded at the end if num_particles < 6

        y: particle of interest, future timesteps
            (num_samples x windowsize/2 x 5)
            last dimension: [px, py, vx, vy, mass]
--]]
function dataloader:next_batch()
    self.current_batch = self.current_batch + 1
    if self.current_batch > self.nbatches then self.current_batch = 1 end

    local minibatch_data = self.dataset[self.configs[self.batch_idxs[self.current_batch]]]
    local minibatch_p = minibatch_data.particles  -- (num_samples x num_particles x windowsize x 5)
    local minibatch_g = minibatch_data.goos  -- (num_samples x num_goos x 5)
    local minibatch_m = minibatch_data.mask  -- 5
    local this_particles, other_particles = expand_for_each_particle(minibatch_p)
    local num_samples, windowsize = unpack(torch.totable(this_particles:size()))
    local num_particles = minibatch_p:size(2)

    -- expand goos to number of examples. m_goos stays constant anyways
    local m_goos = {}
    for i=1,num_particles do m_goos[#m_goos+1] = minibatch_g end
    m_goos = torch.cat(m_goos,1)
    
    -- pad other_particles with 0s
    assert(minibatch_m[minibatch_m:eq(0)]:size(1) == 5 - other_particles:size()[2])  -- make sure we are padding the right amount
    local num_to_pad = #torch.totable(minibatch_m[minibatch_m:eq(0)])
    if num_to_pad > 0 then 
        local pad_p = torch.Tensor(num_samples, num_to_pad, windowsize, 5):fill(0)
        other_particles = torch.cat(other_particles, pad_p, 2)
    end

    -- split into x and y
    local num_past = math.floor(windowsize/2)
    local this_x = this_particles[{{},{1,num_past},{}}]
    local others_x = other_particles[{{},{},{1,num_past},{}}]
    local y = this_particles[{{},{num_past+1,-1},{}}]

    -- assert num_samples are correct
    assert(this_x:size(1) == num_samples and others_x:size(1) == num_samples and 
            m_goos:size(1) == num_samples and y:size(1) == num_samples)
    -- assert number of axes of tensors are correct
    assert(this_x:size():size(1)==3 and others_x:size():size(1)==4 and 
            y:size():size(1)==3)
    -- assert seq length is correct
    assert(this_x:size(2)==num_past and others_x:size(3)==num_past and 
            y:size(2)==num_past)
    -- check padding
    assert(others_x:size(2)==5)
    -- check data dimension
    assert(this_x:size(3) == 5 and others_x:size(4) == 5 and y:size(3) == 5)

    -- cuda
    if common_mp.cuda then
        this_x          = this_x:cuda()
        others_x        = others_x:cuda()
        m_goos          = m_goos:cuda()
        minibatch_m     = minibatch_m:cuda()
        y               = y:cuda()
    end
    return {this=this_x, others=others_x, goos=m_goos, mask=minibatch_m, y=y}
end

-- return dataloader

d = dataloader.create('trainset','/om/user/mbchang/physics-data/dataset_files',false)
-- d = dataloader.create('trainset','hey',false)
print(d:next_batch())

