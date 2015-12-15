local T = require 'pl.tablex'

local pc = true
local SEQ_LENGTH = 10

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

function create_experiment_string(keys, params)
    local foldername = 'results'
    for i=1,#keys do foldername = foldername .. '_'..keys[i]..'='..params[keys[i]] end
    return foldername..'debug_curriculum'
end


personal_mp = {
    batch_size  = 1,
    seq_length  = SEQ_LENGTH,  -- max other objects
    winsize     = 10,
    max_epochs  = 10, 
    dataset_folder = 'hey',
    num_threads = 1,
    cuda        = false,
    cunn        = false
}

openmind_mp = {
    batch_size = 100,
    seq_length = SEQ_LENGTH,  
    winsize    = 20,
    max_epochs = 10, 
    dataset_folder = '/om/user/mbchang/physics-data/dataset_files',
    num_threads = 4,
    cuda       = true,
    cunn       = true
}

-- Common parameters
common_mp = {
    layers        = 4,
    rnn_dim       = 100,
    rand_init_wts = false,
    seed          = 123
}

if pc then 
    common_mp = merge_tables(common_mp, personal_mp)
else
    common_mp = merge_tables(common_mp, openmind_mp)
end

common_mp.input_dim = 8.0*common_mp.winsize/2
common_mp.out_dim = 8.0*common_mp.winsize/2
common_mp.results_folder = create_experiment_string({'batch_size', 'seq_length', 'layers', 'rnn_dim', 'max_epochs'}, common_mp)

-- Training parameters
train_mp = merge_tables(common_mp, {
      shuffle               = true,
      curriculum            = false,
      max_grad_norm         = 10,

      -- Data Logging Parameters
      save_every            = 50,  -- save every fifty iters
      print_every           = 10   -- print every ten iters
})


-- Testing parameters
test_mp = merge_tables(common_mp, {      
      shuffle               = true,
})


