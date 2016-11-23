require 'nn';
require 'rnn';

-- local steps = 3
-- local max_obj = 4
-- local obj_dim = 5
-- local bsize = 2
-- local layers = 1

-- local input_table = {}
-- for i = 1,steps do 
--     local input = torch.rand(bsize, max_obj, obj_dim)
--     local row = math.random(max_obj-1)
--     print(row)
--     input[{{},{max_obj-row+1,-1},{}}]:zero()
--     table.insert(input_table, input)
-- end

-- for i = 1,steps do
--     print(input_table[i])
-- end

local core = nn.Sequential()
-- for i=1,layers do
--     local layer = nn.Sequential()
--     layer:add(nn.Reshape(max_obj*obj_dim,true))
--     layer:add(nn.Linear(max_obj*obj_dim, max_obj*obj_dim))
--     -- you can do a zero right here
--     layer:add(nn.Reshape(max_obj,obj_dim,true))
--     core:add(nn.MaskZero(layer,2))
-- end
-- model = core
-- -- model = nn.Sequencer(core)

-- local output_table = model:forward(input_table[1])

-- print('>>>>')
-- print(output_table[1])

-- -- for i = 1, steps do
-- --     print(output_table[i])
-- -- end

-- print('----')
-- -- print(model)
-- -- print(model.modules[1].modules[1].modules[2].modules[1].modules[3].output)
-- -- print(model.modules[1].modules[1].modules[2].modules[1].modules[3].gradInput)

-- local a = torch.rand(3,4,2)
-- print(a)
-- a[{{},{3},{}}]:zero()
-- print(a)

local b = torch.rand(3,8)
print(b)
b[{{3},{}}]:zero()
print(b)

local n = nn.Linear(8,8)
local n2 = nn.Sequential()
n2:add(nn.Reshape(8,trues))
n2:add(n)
local m = nn.MaskZero(n,1)
local m2 = nn.MaskZero(n2,1)
print(m:forward(b))
print(n:forward(b))
print(m2:forward(b))
-- it doesn't work for nn.Reshape because I reshape it!

-- local nn = nn.Sequential()

-- what if I split it first: so I have a table of batches of (max_obj, obj_dim)
-- you have to remember to share the LSTM within the layer though! (clone it because otherwise the gradients might not propagate correctly)


-- I can just multiply by a mask then.

