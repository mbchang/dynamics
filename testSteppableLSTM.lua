require 'torch'
require 'metaparams'
require 'nn'
-- require 'model'
require 'SteppableLSTM'

torch.manualSeed(123)

local input_dimension = 100
local output_dimension = 50
local hidden_dimension = 50
local num_layers = 1
local dropout = false

local DataLoader = require 'DataLoader'
local loader = DataLoader.create('data_small', train_mp.seq_length, train_mp.batch_size, train_mp.shuffle)

-- local encoder = init_baseline_encoder(train_mp, input_dimension)
local lstm = nn.SteppableLSTM(train_mp, input_dimension, output_dimension, hidden_dimension, num_layers, dropout)
-- local decoder = init_baseline_decoder(train_mp, output_dimension)

local x, y = loader:next_batch(train_mp)
for t = 1, train_mp.seq_length do 
    -- local encodings = encoder:forward(torch.squeeze(x[{{},{t}}]))
    -- local lstm_out = lstm:step(encodings)
    -- local predictions = decoder:forward(lstm_out)

    local predicitons = lstm:step(torch.squeeze(x[{{},{t}}]))
    print(predictions)
end




