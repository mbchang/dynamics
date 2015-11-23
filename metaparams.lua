local T = require 'pl.tablex'

local pc = true

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
    return foldername
end


personal_mp = {
    batch_size  = 1,
    seq_length  = 10,
    max_epochs  = 100, 
    cuda        = false
}

openmind_mp = {
    batch_size = 100,
    seq_length = 10,
    max_epochs = 20, 
    cuda       = false
}

-- Common parameters
common_mp = {
    layers        = 4,
    input_dim     = 4*10, -- winsize/2 is 10
    rnn_dim       = 100,
    out_dim       = 4*10, -- winsize/2 is 10
    cudnn         = false,
    rand_init_wts = false,
    seed          = 123
}

if pc then 
    common_mp = merge_tables(common_mp, personal_mp)
else
    common_mp = merge_tables(common_mp, openmind_mp)
end

-- common_mp.dataset_folder = '/om/user/mbchang/physics-data/dataset_files'
common_mp.dataset_folder = 'hey'
common_mp.results_folder = create_experiment_string({'batch_size', 'seq_length', 'layers', 'rnn_dim'}, common_mp)

-- Training parameters
train_mp = merge_tables(common_mp, {
      shuffle               = true,
      max_grad_norm         = 10,

      -- Data Logging Parameters
      save_every            = 50,  -- save every fifty iters
      print_every           = 10   -- print every ten iters
})


-- Testing parameters
test_mp = merge_tables(common_mp, {      
      shuffle               = true,
})


