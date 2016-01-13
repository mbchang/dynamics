require 'torch'
require 'metaparams'
require 'gnuplot'
require 'paths'
require 'utils'

function plot_train_losses(losses_file, plotfolder)
    -- NOTE THAT THIS IS ACROSS ALL EPOCHS
    if not paths.dirp(plotfolder) then
        paths.mkdir(plotfolder)
    end

    local data = torch.load(losses_file..'.t7')
    local train_losses = torch.log(torch.Tensor(data.losses))
    local grad_norms = torch.Tensor(data.grad_norms)

    local skipped_losses = {}
    local x = torch.range(1, train_losses:size(1), 100)
    for i=1,train_losses:size(1),100 do
        skipped_losses[#skipped_losses+1] = train_losses[i]
    end
    skipped_losses = torch.Tensor(skipped_losses)

    -- local plotfile =plotfolder..'/'..losses_file..'_logskippedlosses.png'
    local plotfile = losses_file..'_logskippedtrainlosses.png'

    gnuplot.pngfigure(plotfile)
    gnuplot.xlabel('batch')
    gnuplot.ylabel('Log MSE Loss')
    gnuplot.title('Losses On Training Set')  -- change
    gnuplot.plot({x, skipped_losses})
    gnuplot.plotflush()
end

function plot_experiment_results(experiment_results_file, plotfolder)
    if not paths.dirp(plotfolder) then paths.mkdir(plotfolder) end

    local experiment_results = torch.load(experiment_results_file)
    local train_results = {}
    local dev_results = {}
    local all_results = {}
    for learning_rate, losses in pairs(experiment_results) do
        train_results[#train_results+1] = {'train '.. learning_rate, torch.log(losses.results_train_losses)}
        dev_results[#dev_results+1] = {'dev '.. learning_rate, torch.log(losses.results_dev_losses)}
        all_results[#all_results+1] = {'train '.. learning_rate, torch.log(losses.results_train_losses)}
        all_results[#all_results+1] = {'dev '.. learning_rate, torch.log(losses.results_dev_losses)}
    end

    -- gnuplot.pngfigure(plotfolder..'/traintest.png')
    -- gnuplot.xlabel('epoch')
    -- gnuplot.ylabel('Log MSE Loss')
    -- gnuplot.title('Losses On Training Set')
    -- gnuplot.plot(unpack(train_results))
    -- gnuplot.plotflush()
    --
    -- gnuplot.pngfigure(plotfolder..'/testest.png')
    -- gnuplot.xlabel('epoch')
    -- gnuplot.ylabel('Log MSE Loss')
    -- gnuplot.title('Losses On Development Set')
    -- gnuplot.plot(unpack(dev_results))
    -- gnuplot.plotflush()

    gnuplot.pngfigure(plotfolder..'/alltest.png')
    gnuplot.xlabel('epoch')
    gnuplot.ylabel('Log MSE Loss')
    gnuplot.title('Losses')
    gnuplot.plot(unpack(all_results))
    gnuplot.plotflush()
end

function plot_all_experiments(parent_folder, plotfn, logfile)
    for experiment_folder in paths.iterdirs(parent_folder) do
        local plot_folder = paths.concat(parent_folder, experiment_folder)
        -- local er_file = plot_folder..'/experiment_results.t7'
        -- plot_experiment_results(er_file, plot_folder)
        local er_file = plot_folder..'/'..logfile
        plotfn(er_file, plot_folder)
    end
end


function plot_all_training(parent_folder)
    local results_files = {'losses,lr=0.0005_results',
                            'losses,lr=5e-05_results',
                            'losses,lr=5e-06_results'}
    for _, train_results in pairs(results_files) do

        try {
            plot_all_experiments(parent_folder, plot_train_losses, train_results),
        catch {
              function(error)
                 print('caught error: ' .. error)
              end
           }
        }
    end
end



-- losses,lr=0.0005_results.t7  losses,lr=5e-06_results.t7   saved_model,lr=5e-05.t7
-- experiment_results.t7        losses,lr=5e-05_results.t7

-- assert(false)
-- common_mp.results_folder = 'rand_order_results_batch_size=100_seq_length=10_layers=4_rnn_dim=100'
-- print(common_mp.results_folder)

-- plot_experiment_results(common_mp.results_folder .. '/experiment_results.t7', 'plots')
-- plot_train_losses(common_mp.results_folder .. '/losses,lr=5e-05_results', '.')


plot_all_training('openmind')
-- plot_all_experiments('openmind', plot_experiment_results, 'experiment_results.t7')
