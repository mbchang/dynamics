require 'nn'
require 'rnn'
require 'nngraph'
nngraph.setDebug(true)


local num_layers = 2
local num_obj = 3
local obj_dim = 6
local in_dim = 6
local hid_dim = 4
local out_dim = 6
local bsize = 2

local nop_core_input = {}
for i=1,num_obj do
    table.insert(nop_core_input, torch.rand(bsize, obj_dim))
end

-- in: table of length num_obj of (bsize, obj_dim)
-- out: table of length num_obj of (bsize, out_dim)
function create_nop_core(in_dim, hid_dim, num_layers)
    local object_core = nn.Sequential()
    if num_layers == 1 then
        object_core:add(nn.Linear(in_dim, hid_dim, bias))
    else
        for i = 1, num_layers do -- TODO make sure this is comparable to encoder decoder architecture in terms of layers
            if i == 1 then 
                object_core:add(nn.Linear(in_dim, hid_dim, bias))
                object_core:add(nn.ReLU())
            elseif i == num_layers then
                object_core:add(nn.Linear(hid_dim, hid_dim, bias))  -- don't have any zero output
            else
                object_core:add(nn.Linear(hid_dim, hid_dim, bias))
                object_core:add(nn.ReLU())
            end
        end
    end
    object_core:add(nn.Reshape(bsize, 1, hid_dim))
    local object_cores = nn.Sequencer(object_core) -- produces a table of num_obj of {(bsize, hid_dim)}
    local object_core_net = nn.Sequential()
    object_core_net:add(object_cores)
    object_core_net:add(nn.JoinTable(2))  -- (bsize, num_obj, hid_dim)
    return object_core_net
end

local object_cores = create_nop_core(in_dim, hid_dim, num_layers)
local object_cores_out = object_cores:forward(nop_core_input)

-- print(nop_core_input)
-- print(object_cores_out)

-- for i=1,num_obj do
--     print(nop_core_input[i])
--     print(object_cores.modules[1].output[i])
-- end


-----------------------------------------------------------------
local nop_decoder_input = {}
for i=1,num_obj do
    table.insert(nop_decoder_input, torch.rand(bsize, 2*hid_dim))
end

-- in: table of length num_obj of (bsize, 2*hid_dim)
-- out: table of length num_obj of (bsize, out_dim)
function create_nop_decoder(hid_dim, out_dim, num_layers)
    local decoder = nn.Sequential()
    if num_layers == 1 then
        decoder:add(nn.Linear(2*hid_dim, out_dim))
    else
        for i = 1, num_layers do -- TODO make sure this is comparable to encoder decoder architecture in terms of layers
            if i == 1 then 
                decoder:add(nn.Linear(2*hid_dim, hid_dim))
                decoder:add(nn.ReLU())
            elseif i == num_layers then
                decoder:add(nn.Linear(hid_dim, out_dim))
            else
                decoder:add(nn.Linear(hid_dim, hid_dim))
                decoder:add(nn.ReLU())
            end
        end
    end
    local decoders = nn.Sequencer(decoder)
    return decoders
end

local object_decoders = create_nop_decoder(hid_dim, out_dim, num_layers)
local object_decoders_out = object_decoders:forward(nop_decoder_input)

-- print(nop_decoder_input)
-- print(object_decoders_out)

-- for i=1,num_obj do
--     print(nop_decoder_input[i])
--     print(object_decoders_out[i])
-- end


-----------------------------------------------------------------

local nop_core_output = torch.rand(bsize, num_obj, hid_dim)
local nbrhd_masks = {}
for i=1,num_obj do
    -- table of length num_obj of tensor (bsize, num_obj, hid_dim)
    -- binary matrix (bsize, num_obj) and repeat it across hid_dim
    local nbrhd_mask = torch.repeatTensor(torch.rand(bsize,num_obj,1):le(0.5),1,1,hid_dim):double()
    table.insert(nbrhd_masks, nbrhd_mask)
end

function create_nbrhd_masker(bsize, hid_dim)
    local nop_core_output_node = nn.Identity()()
    local nbrhd_mask_node = nn.Identity()()  -- tensor (bsize, num_obj, hid_dim)
    local nbrhd_masks_node = nn.Identity()()

    local masked_object_encodings_tensor = nn.Identity()(nbrhd_mask_node)
    -- local masked_object_encodings_tensor = nn.CMulTable(){nbrhd_mask_node,nbrhd_mask_node}  -- tensor (bsize, num_obj, hid_dim)
    local reduced_masked_object_encodings_tensor = nn.Sum(2)(masked_object_encodings_tensor)  -- tensor (bsize, hid_dim)
    local reshaped_reduced_masked_object_encodings_tensor = nn.Reshape(bsize, 1, hid_dim)(reduced_masked_object_encodings_tensor)  -- tensor (bsize, 1, hid_dim)
    local masker_core = nn.gModule({nbrhd_mask_node},{reshaped_reduced_masked_object_encodings_tensor})

    local masker_sequencer = nn.Sequencer(masker_core)  -- outputs table of length num_obj of tensors (bsize, 1, hid_dim)
    local masker_sequencer_out = masker_sequencer(nbrhd_masks_node)  -- you need to somehow repeat nop_core_output_node
    local joined_masker_sequencer_out = nn.JoinTable(2)(masker_sequencer_out)   -- tensor (bsize, num_obj, hid_dim)    

    local nbhrd_masker_net = nn.gModule({nbrhd_masks_node},{joined_masker_sequencer_out})

    -- nbhrd_masker_net.name = 'hey'

    -- os.execute('open -a  Safari hey.svg')
    return nbhrd_masker_net
end



-- -- concatenate the output of masker with reshaped_object_encodings
-- -- both should be tensors tensor (bsize, num_obj, hid_dim)

-- local object_decoder_input_tensor = nn.JoinTable(3){object_encodings_tensor,joined_masker_sequencer_out}  -- tensor (bsize, num_obj, 2*hid_dim)

-- -- break into table
-- local object_decoder_input_table = nn.SplitTable(2)(object_decoder_input_tensor)

local nbrhd_masker = create_nbrhd_masker(bsize, hid_dim)
-- local nbrhd_masker_out = nbrhd_masker:forward({nop_core_output, nbrhd_masks})

print({nop_core_output, nbrhd_masks})


