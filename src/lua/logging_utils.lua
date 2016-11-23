local T = require 'pl.tablex'

function merge_tables(t1, t2)
    -- Merges t2 and t1, overwriting t1 keys by t2 keys when applicable
    local merged_table = T.deepcopy(t1)
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
    for i=1,#keys do
        foldername = foldername .. '_'..keys[i]..'='..params[keys[i]]
    end
    return foldername
end
