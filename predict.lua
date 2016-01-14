require 'utils'

local Tester = require 'test'
--
-- local t =  Tester.create('testset', test_mp)  -- will choose an example based on random shuffle or not
-- -- local model = 'results_batch_size=100_seq_length=10_layers=2_rnn_dim=100_max_epochs=20/saved_model,lr=0.0005.t7'
-- -- local model = 'results_batch_size=1_seq_length=10_layers=4_rnn_dim=100_max_epochs=10debug_curriculum/saved_model,lr=0.00025.t7'
--
-- -- local model = 'pc/results_batch_size=1_seq_length=10_layers=4_rnn_dim=100_max_epochs=50trainloss_gt_devloss/saved_model,lr=0.0005.t7'
--
-- local model = 'logs/results_batch_size=100_seq_length=10_layers=2_rnn_dim=100_max_epochs=20floatnetwork/saved_model,lr=0.0005.t7'
--
-- t:test(model, 1, true)

function predict(parent_folder, experiment_folder, logfile)
    -- hacky
    local start = experiment_folder:find('layers=') + #'layers='
    local finish = experiment_folder:find('_rnn_dim=') - 1
    local num_layers = tonumber(experiment_folder:sub(start, finish))
    -- cannot do common_mp.layers! because test_mp has already been constructed
    -- when you required 'test'
    if num_layers == 4 then
        test_mp.layers = 4
    else
        test_mp.layers = 2
    end
    local t = Tester.create('testset', test_mp)

    local model = parent_folder..'/'..experiment_folder..'/'..logfile
    t:test(model,1,true)
    if common_mp.cuda then cutorch.synchronize() end
    collectgarbage()
end


function predict_all_experiments(parent_folder, logfile)
    for experiment_folder in paths.iterdirs(parent_folder) do
        if experiment_folder:sub(1,#'results') == 'results' then
            local files = paths.dir(parent_folder..'/'..experiment_folder)
            for _, file in pairs(files) do
                if file == logfile then
                    predict(parent_folder, experiment_folder, logfile)
                end
            end
        end
    end
end


function predict_all_models(parent_folder)
    local saved_models = {'saved_model,lr=0.0005.t7',
                            'saved_model,lr=5e-05.t7',
                            'saved_model,lr=5e-06.t7'}
    for _, saved_model in pairs(saved_models) do
        predict_all_experiments(parent_folder, saved_model)
    end
end

predict_all_models('logs')
