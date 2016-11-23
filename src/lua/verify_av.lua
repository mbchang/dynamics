require 'json_interface'

local function iter_files_ordered(folder)
    -- Wrote to ../data/mixed_n6_t60_ex50000_rd/jsons/mixed_n6_t60_ex50000_rd_chksize100_188.json
    -- Wrote to ../data/mixed_n6_t60_ex50000_rd/jsons/mixed_n6_t60_ex50000_rd_chksize100_189.json
    -- Wrote to ../data/mixed_n6_t60_ex50000_rd/jsons/mixed_n6_t60_ex50000_rd_chksize100_190.json
    -- Wrote to ../data/mixed_n6_t60_ex50000_rd/jsons/mixed_n6_t60_ex50000_rd_chksize100_191.json
    -- Wrote to ../data/mixed_n6_t60_ex50000_rd/jsons/mixed_n6_t60_ex50000_rd_chksize100_192.json
    local files = {}
    for f in paths.iterfiles(folder) do
        table.insert(files, f) -- good
    end
    table.sort(files)  -- mutates files
    return files
end

local function count_examples(jsonfolder)
    local ordered_files = iter_files_ordered(jsonfolder)
    local num_examples = 0
    for _, jsonfile in pairs(ordered_files) do
        local data = load_data_json(paths.concat(jsonfolder,jsonfile))  -- (num_examples, num_obj, num_steps, object_raw_dim)
        local av = data[{{},{},{},{6}}]
        print(av:gt(math.pi):nonzero())
        print(av:lt(-math.pi):nonzero())
        print('--')
        -- print(data:size())
    end
    collectgarbage()
    return num_examples
end

count_examples('/om/user/mbchang/physics/data/tower_n6_t120_ex25000_rda/jsons')