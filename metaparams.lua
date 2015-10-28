T = require 'pl.tablex'

pc = true

function merge_tables(t1, t2) 
    -- Merges t2 and t1, overwriting t1 keys by t2 keys when applicable
    merged_table = T.deepcopy(t1)
    for k,v in pairs(t2) do 
        -- if merged_table[k] then
        --     error('t1 and t2 both contain the key: ' .. k)
        -- end
        merged_table[k]  = v 
    end 
    return merged_table
end

-- Common parameters
common_mp = {
    -- Convolutional Parameters
    frame_height          = 150,
    frame_width           = 150,
    feature_maps          = 96,  -- 96 on openmind, 16 on local
    color_channels        = 3,
    encoder_filter_size   = 5,
    decoder_filter_size   = 7,
    pool_size             = 2,
    cnn_last_dim          = 15, -- if input is (150,150), filter is (5,5), pool is (2,2) and we do 3 convolutions for encoder, 4 convolutions for decoder

    -- LSTM Parameters
    LSTM_input_dim        = 100,
    LSTM_hidden_dim       = 50,
    LSTM_output_dim       = 100,
    layers                = 1,  -- not implemented yet

    -- CUDA
    cuda                  = false,
    cudnn                 = false,
    rand_init_wts         = false
}

-- Logging parameters
common_mp.results_folder = 'results_featmaps='..common_mp.feature_maps..'_seqlen=3'

-- Training parameters
train_mp = merge_tables(common_mp, {
      batch_size            = 100,  -- 100 on openmind
      seq_length            = 3,  -- Something like 2 in, 1 out
      shuffle               = false,
      max_grad_norm         = 10,
      max_epochs            = 20,  -- 100 on openmind
      seed                  = 123,

      -- Data Logging Parameters
      save_every            = 50,  -- save every fifty iters
      print_every           = 10   -- print every ten iters
})

-- Training parameters to ignore
train_mp_ignore = {
    frame_height            = true,
    frame_width             = true,
    color_channels          = true,
    save_every              = true,
    print_every             = true
}

-- Testing parameters
test_mp = merge_tables(common_mp, {      
      batch_size            = 100,
      seq_length            = 3,
      shuffle               = true,
      seed                  = 234
})

personal_mp = {
    feature_maps = 16,
    batch_size = 3,
    seq_length = 2,
    max_epochs = 3
}

openmind_mp = {
    feature_maps = 32,
    batch_size = 100,
    seq_length = 5,
    max_epochs = 20
}

if pc then 
    train_mp = merge_tables(train_mp, personal_mp)
    test_mp = merge_tables(test_mp, personal_mp)
else
    train_mp = merge_tables(train_mp, openmind_mp)
    test_mp = merge_tables(test_mp, openmind_mp)  
end


