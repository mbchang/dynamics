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
    print(examples)
    return examples
end


function dataloader.create(dataset_name, dataset_folder, shuffle)
    --[[
        Input
            dataset_file: file containing data
            batch_size: batch size
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
    self.current_batch = 1

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
    local num_samples, num_particles, windowsize = unpack(minibatch_p:size)
    local this_particles = {}
    local other_particles = {}
    for p=1:num_samples do
        this = batch_particles[{{},{i},{},{}}]  -- (num_samples x windowsize x 5)
        other = torch.cat(batch_particles[{{},{1,i-1},{},{}}], 
                        batch_particles[{{},{i+1,-1},{},{}}], 2)  -- leave this particle out (num_samples x (num_particles-1) x windowsize x 5)

        this_particles[#this_particles+1] = this
        other_particles[#other_particles+1] = other
    end

    this_particles = torch.cat(this_particles,1)  -- concatenate along batch dimension
    other_particles = torch.cat(other_particles,1)
    return this_particles, other_particles
end


function dataloader:next_batch()
    self.current_batch = self.current_batch + 1
    if self.current_batch > self.nbatches then self.current_batch = 1 end

    local minibatch_data = self.dataset[self.configs[self.batch_idxs[current_batch]]]
    local minibatch_p = minibatch_data.particles  -- (num_samples x num_particles x windowsize x 5)
    local minibatch_g = minibatch_data.goos  -- (num_samples x num_goos x 5)
    local minibatch_m = minibatch_data.mask  -- 5
    local num_samples, num_particles, windowsize = unpack(minibatch_p:size)
    local this_particles, other_particles = expand_for_each_particle(minibatch_p)

    -- split into x and y
    local num_past = math.floor(windowsize/2)
    local this_x = this_particles[{{},{},{1,num_past},{}}]
    local others_x = other_particles[{{},{},{1,num_past},{}}]
    local y = this_particles[{{},{},{num_past+1,-1},{}}]

    -- cuda
    if common_mp.cuda then
        this_x      = this_x:cuda()
        others_x    = others_x:cuda()
        goos        = goos:cuda()
        mask        = mask:cuda()
        y           = y:cuda()
    end

    return {this_x, others_x, goos, mask, y}
end

return dataloader

