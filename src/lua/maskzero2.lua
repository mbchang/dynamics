require 'nn';
require 'nngraph'
require 'rnn';
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(123)

local steps = 3
local max_obj = 4
local obj_dim = 5
local bsize = 2
local layers = 1

local data_table = {}
for i = 1,steps do 
    local input = torch.rand(bsize, max_obj, obj_dim)
    print(input)
    local row = math.random(max_obj-1)
    print(row)
    input[{{},{max_obj-row+1,-1},{}}]:zero()
    print(input)
    print('.....')
    local mask = input:gt(0):float()
    input = input:reshape(bsize, max_obj*obj_dim)
    mask = mask:reshape(bsize, max_obj*obj_dim)
    table.insert(data_table, {input, mask})
end

for i = 1,steps do
    print(data_table[i])
end

function masker(layertype, activation)
    local layer_in = nn.Identity()()
    local mask = nn.Identity()()
    local lin_out = layertype(layer_in)
    local lin_act_out = activation(lin_out)
    local mask_out = nn.CMulTable(){lin_act_out, mask}
    local layer = nn.gModule({layer_in, mask}, {mask_out, mask})
    return layer
end

-- local core_in = nn.Identity()
-- local core_mask = nn.Identity()
local core = nn.Sequential()
for i=1,layers do
    local layertype = nn.Linear(max_obj*obj_dim, max_obj*obj_dim)
    local activation = nn.ReLU()
    local layer = masker(layertype, activation)
    core:add(layer)
end
core:add(nn.CMulTable())  -- get rid of the mask

-- -- model = core
model = nn.Sequencer(core)

local output_table = model:forward(data_table)

print('>>>>>>>>>')
for i = 1,steps do
    -- print(output_table[i][1])
    -- print(output_table[i][2])  -- this is the mask!
    print(output_table[i])
end
