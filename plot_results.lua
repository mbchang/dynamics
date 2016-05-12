require 'torch'
--require 'metaparams'
require 'gnuplot'
require 'paths'
require 'utils'

pp = lapp[[
   -i,--infolder           (default "in")           folder to read
]]

-- will have to change this soon
function read_log_file(logfile)
    local data = {}
    for line in io.lines(logfile) do
        data[#data+1] = tonumber(line) --ignores the string at the top
    end
    data = torch.Tensor(data)
    return data
end

-- will have to change this soon
function read_log_file_3vals(logfile)
    local data1 = {}
    local data2 = {}
    local data3 = {}
    for line in io.lines(logfile) do
        local x = filter(function(x) return not(x=='') end,
                            stringx.split(line:gsub("%s+", ","),','))
        data1[#data1+1] = tonumber(x[1]) --ignores the string at the top
        data2[#data2+1] = tonumber(x[2]) --ignores the string at the top
        data3[#data3+1] = tonumber(x[3]) --ignores the string at the top
    end
    local data = torch.cat({torch.Tensor(data1), torch.Tensor(data2), torch.Tensor(data3)}, 2)
    return data
end

-- info{outfilename, xlabel, ylabel, title}, all strings
-- like{outfilename, 'batch', 'Log MSE Loss', 'Losses On Training Set'}
function plot_tensor(tensor, info, subsamplerate)
    local toplot = subsample(tensor, subsamplerate)
    gnuplot.pngfigure(info[1])
    gnuplot.xlabel(info[2])
    gnuplot.ylabel(info[3])
    gnuplot.title(info[4])  -- change
    gnuplot.plot(unpack(toplot))
    gnuplot.plotflush()
end

function subsample1(tensor, rate)
    local subsampled = {}
    local x = torch.range(1, tensor:size(1), rate)
    for i=1,tensor:size(1),rate do
        subsampled[#subsampled+1] = tensor[i]
    end
    subsampled = torch.Tensor(subsampled)
    return subsampled
end

function subsample(tensor, rate)
    if tensor:dim() == 1 then
        return {subsample1(tensor, rate), '~'}
    else  -- more than one variable
        local y = map(function (x) return subsample1(torch.Tensor(x), rate) end,
                      torch.totable(tensor:t()))
        return {{'train', y[1],'~'},{'val', y[2],'~'}, {'test', y[3],'~'}}  -- hardcoded
    end
end

-- for main.lua
function plot_training_losses(logfile, savefile)
    local data = read_log_file(logfile)
    local subsamplerate = 10
    plot_tensor(data,
                {savefile,
                 'batch (every '..subsamplerate..')',
                 'Log Euclidean Distance',
                 'Losses On Training Set'},
                 subsamplerate)
end

function plot_experiment(logfile, savefile)
    local data = read_log_file_3vals(logfile)
    local subsamplerate = 1
    plot_tensor(data,
                {savefile,
                 'Epoch',-- (every '..subsamplerate..')',
                 'Log Euclidean Distance',
                 'Losses'},
                 subsamplerate)
 end

-- losses,lr=0.0005_results.t7  losses,lr=5e-06_results.t7   saved_model,lr=5e-05.t7
-- experiment_results.t7        losses,lr=5e-05_results.t7

-- assert(false)
-- common_mp.results_folder = 'rand_order_results_batch_size=100_seq_length=10_layers=4_rnn_dim=100'
-- print(common_mp.results_folder)

-- plot_experiment_results(common_mp.results_folder .. '/experiment_results.t7', 'plots')
-- plot_train_losses(common_mp.results_folder .. '/losses,lr=5e-05_results', '.')


-- plot_all_training('openmind')
-- plot_all_experiments('pc', plot_experiment_results, 'experiment_results.t7')



-- -- read_log_file('openmind/baselinesubsampled_opt_adam_lr_0.0005')
-- plot_training_losses('/Users/MichaelChang/Documents/Researchlink/SuperUROP/Code/dynamics/oplogs/baselinesubsampled_opt_adam_lr_0.001/train.log',
-- --                      'hihhihhihih')
-- plot_experiment('/Users/MichaelChang/Documents/Researchlink/SuperUROP/Code/'..
--                 'dynamics/oplogs/baselinesubsampledcontig_opt_optimrmsprop_'..
--                 'testcfgs_[:-2:2-:]_traincfgs_[:-2:2-:]_lr_0.001/experiment.log',
--                 'hihhihhihih')

-- plot_training_losses('/Users/MichaelChang/Documents/Researchlink/SuperUROP/Code/dynamics/oplogs/baselinesubsampled_opt_adam_lr_0.001/train.log',
-- --                      'hihhihhihih')

-- print(pp.infolder)
plot_experiment(pp.infolder..'/experiment.log', pp.infolder..'/experiment.png')
plot_training_losses(pp.infolder..'/train.log',pp.infolder..'/train.png')
