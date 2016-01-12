local Tester = require 'test'

local t =  Tester.create('testset', test_mp)  -- will choose an example based on random shuffle or not
local model = 'results_batch_size=100_seq_length=10_layers=2_rnn_dim=100_max_epochs=20/saved_model,lr=0.0005.t7'
-- local model = 'results_batch_size=1_seq_length=10_layers=4_rnn_dim=100_max_epochs=10debug_curriculum/saved_model,lr=0.00025.t7'
t:test(model, 1, true)
