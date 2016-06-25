-- normalize: requires normalizing constants

-- unnormalize

-- permute (we could possibly use this)

-- one hot (masses)

-- subsample range (we could possibly use this)

-- split into train, validation, test

-- group into batches?


-- fields: relative, object_dim

-- also need to split past/future? Or should that be in the computation?

-- note that these functions do NOT mutate state. They just reference fields
require 'torchx'
require 'json_interface'
require 'data_utils'
require 'paths'

local data_process = {}
data_process.__index = data_process


function data_process.create(args)
    local self = {}
    setmetatable(self, data_process)

    self.pnc = args.position_normalize_constant
    self.vnc = args.velocity_normalize_cnstant
    self.relative = args.relative -- bool
    self.masses = args.masses -- {0.33, 1.0, 3.0, 1e30}
    self.rsi = args.raw_state_indices -- {px: 1, py: 2, vx: 3, vy: 4, m: 5, oid: 6}
    self.si = args.state_indices -- {px: {1}, py: {2}, vx: {3}, vy: {4}, m: {5,8}, oid: {9}}
    self.permute_context = args.permute_context  -- bool: if True will expand the dataset, False won't NOTE: not spending my time permuting for now
    self.bsize = args.batch_size

    -- here you can also include have world parameters
    -- not that object id includes whether it is stationary or not

    return self
end

-- trajectories: (num_examples, num_objects, timesteps, [px, py, vx, vy, mass])
function data_process:normalize(unnormalized_trajectories)
    normalized = unnormalized_trajectories:clone()

    local px, py, vx, vy = self.rsi.px, self.rsi.py, self.rsi.vx, self.rsi.vy

    -- normalize position
    normalized[{{},{},{},{px,py}}] = normalized[{{},{},{},{px,py}}]/self.pnc

    -- normalize velocity
    normalized[{{},{},{},{vx,vy}}] = normalized[{{},{},{},{vx,vy}}]/self.vnc

    return normalized
end

function data_process:unnormalize(normalized_trajectories)
    unnormalized = normalized_trajectories:clone()

    local px, py, vx, vy = self.rsi.px, self.rsi.py, self.rsi.vx, self.rsi.vy

    -- normalize position
    unnormalized[{{},{},{},{px,py}}] = unnormalized[{{},{},{},{px,py}}]*self.pnc

    -- normalize velocity
    unnormalized[{{},{},{},{px,py}}] = unnormalized[{{},{},{},{px,py}}]*self.vnc

    return unnormalized
end

function data_process:mass2onehot(mass)
    local index = torch.find(torch.Tensor(self.masses), mass)[1]
    assert(not(index == nil))
    local onehot = torch.zeros(#self.masses)
    onehot[{{index}}]:fill(1)  -- will throw an error if index == nil
    return onehot
end

function data_process:onehot2mass(onehot)
    assert(onehot:sum() == 1 and #torch.find(onehot, 1) == 1)
    return self.masses[torch.find(onehot, 1)]
end

function data_process:mass2onehotall(trajectories)
    local before = trajectories[{{},{},{},{self.rsi.px, self.rsi.vy}}]:clone()
    local after = trajectories[{{},{},{},{self.rsi.m+1,-1}}]:clone()
    local masses = trajectories[{{},{},{},{self.rsi.m}}]:clone()

    local num_ex = masses:size(1)
    local num_obj = masses:size(2)
    local num_steps = masses:size(3)

    -- expand
    masses = torch.repeatTensor(masses, 1, 1, 1, #self.masses)  -- I just want to tile on the last dimension
    masses:resize(num_ex*num_obj*num_steps, #self.masses)

    for row=1,masses:size(1) do
        masses[{{row}}] = self:mass2onehot(masses[{{row},{1}}]:sum())
    end
    masses:resize(num_ex, num_obj, num_steps, #self.masses)

    -- join
    local trajectoriesonehot = torch.cat({before, masses, after}, 4)

    return trajectoriesonehot
end

--[[ Expands the number of examples per batch to have an example per particle
    Input: unfactorized: (num_samples x num_obj x windowsize x 8)
    Output:
        {
            focus: (num_samples, windowsize, 8)
            context: (num_samples x (num_obj-1) x windowsize x 8) or {}
        }
--]]
function data_process:expand_for_each_object(unfactorized)
    local num_samples, num_obj, _, _ = unpack(torch.totable(unfactorized:size()))
    local focus = {}
    local context = {}
    if num_obj > 1 then
        for i=1,num_obj do  -- this is doing it in transpose order
            -- NOTE: the one-hot encoding has 4 values, and if the last value is 1 that means it is the stationary ball!
            local this = unfactorized[{{},{i},{},{}}]  --all of the particles here should be the same
            if this[{{},{},{},{-2}}]:sum() == 0 then -- only do it if the particle is not stationary
                this = this:reshape(this:size(1), this:size(3), this:size(4))  -- (num_samples x windowsize x 8); NOTE that resize gives the wrong answer!

                local other
                if i == 1 then
                    other = unfactorized[{{},{i+1,-1},{},{}}]
                elseif i == num_obj then
                    other = unfactorized[{{},{1,i-1},{},{}}]
                else
                    other = torch.cat(unfactorized[{{},{1,i-1},{},{}}],
                                unfactorized[{{},{i+1,-1},{},{}}], 2)  -- leave this particle out (num_samples x (num_obj-1) x windowsize x 8)
                end

                -- permute here
                assert(this:size()[1] == other:size()[1])
                -- this: (num_samples, winsize, 8)
                -- other: (num_samples, num_other_particles, winsize, 8)
                -- local num_other_particles = other:size(2)
                -- for j = 1, num_other_particles do
                --
                --     local permuted_other = torch.cat(permute(other),1)
                --     assert(permuted_other:size(1) == factorial(num_other_particles))
                --     for k = 1, factorial(num_other_particles) do
                --         focus[#focus+1] = this
                --     end
                --     context[#context+1] = permuted_other
                --
                --     -- focus[#focus+1] = this
                --     -- context[#context+1] = other
                -- end
                focus[#focus+1] = this
                context[#context+1] = other
            end
        end
    else
        local this = unfactorized[{{},{i},{},{}}]
        focus[#focus+1] = torch.squeeze(this,2) -- (num_samples x windowsize x 8)
    end

    focus = torch.cat(focus,1)  -- concatenate along batch dimension

    -- make context into Torch tensor if more than one particle. Otherwise {}
    if next(context) then
        context = torch.cat(context,1)
        assert(focus:size(1) == context:size(1))
        assert(context:size(2) == num_obj-1)
    end

    return focus, context
end

function data_process:split2batches(data)
    local num_examples = data:size(1)
    assert(num_examples % self.bsize == 0)
    return data:clone():chunk(num_examples/self.bsize,1)
end


-- train-val-test: 70-15-15 split
function data_process:split2datasets(examples)
    local num_test = math.floor(#examples * 0.15)
    local num_val = num_test
    local num_train = #examples - 2*num_test

    local test = {}
    local val = {}
    local train = {}

    -- shuffle examples
    local ridxs = torch.randperm(#examples)
    for i = 1, ridxs:size(1) do
        xlua.progress(i, ridxs:size(1))
        local batch = examples[ridxs[i]]
        if i <= num_train then
            table.insert(train, batch)
        elseif i <= num_train + num_val then
            table.insert(val, batch)
        else
            table.insert(test, batch)
        end
    end
    return {train=train, val=val, test=test}
end

function data_process:save_batches(datasets, savefolder)
    if not paths.dirp(savefolder) then paths.mkdir(savefolder) end
    for k,v in pairs(datasets) do
        local dataset_folder = savefolder..'/'..k
        if not paths.dirp(dataset_folder) then paths.mkdir(dataset_folder) end
        for i=1,#v do
            local batch_file = dataset_folder..'/batch'..i
            torch.save(batch_file,v[i])
        end
    end
end

-- main
function data_process:create_datasets()
    -- each example is a (focus, context) pair
    local json_file = '/Users/MichaelChang/Documents/Researchlink/SuperUROP/Code/physics_worlds/balls.json'
    local data = load_data_json(json_file)
    data = self:normalize(data)
    data = self:mass2onehotall(data)
    local focus, context = self:expand_for_each_object(data)-- TODO include object id in expand for each object
    local focus_batches = self:split2batches(focus)
    local context_batches = self:split2batches(context)
    local all_batches = {}
    for b=1,#focus_batches do
        table.insert(all_batches, {focus_batches[b], context_batches[b]})
    end
    local datasets = self:split2datasets(all_batches)
    self:save_batches(datasets, 'debug')
end

-- to be compatible with previous code, for now
function regression_interface(datasets)
    -- just need the mask as well as split into past and future
end

-- th> a
-- {
--   1 : FloatTensor - size: 50x2x9
--   2 : FloatTensor - size: 50x10x2x9
--   3 : FloatTensor - size: 50x2x9
--   4 : FloatTensor - size: 10
--   5 : "worldm5_np=2_ng=0_slow"
--   6 : 1
--   7 : 50
--   8 : FloatTensor - size: 50x10x2x9
-- }

-- return data_process

args = {position_normalize_constant=800,
        velocity_normalize_cnstant=50,
        relative=true,
        masses={0.33, 1.0, 3.0, 1e30},
        raw_state_indices={px=1, py=2, vx=3, vy=4, m=5, oid=6},
        state_indices={px={1}, py={2}, vx={3}, vy={4}, m={5,8}, oid={9}},
        permute_context=False,
        batch_size=4}


dp = data_process.create(args)
dp:create_datasets()
