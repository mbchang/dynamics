require 'torch'
require 'metaparams'
require 'gnuplot'
require 'paths'

function plot_train_losses(losses_file, plotfolder)
    -- NOTE THAT THIS IS ACROSS ALL EPOCHS
    if not paths.dirp(plotfolder) then 
        paths.mkdir(plotfolder)
    end

    local data = torch.load(losses_file)
    local train_losses = torch.log(torch.Tensor(data.losses))
    local grad_norms = torch.Tensor(data.grad_norms)

    gnuplot.pngfigure(plotfolder..'/trainlosses.png')
    gnuplot.xlabel('batch   ')
    gnuplot.ylabel('Log BCE Loss')
    gnuplot.title('Losses On Training Set')
    gnuplot.plot(train_losses)
    gnuplot.plotflush()
end

function plot_experiment_results(experiment_results_file, plotfolder)
    if not paths.dirp(plotfolder) then 
        paths.mkdir(plotfolder)
    end

    local experiment_results = torch.load(experiment_results_file)
    local train_results = {}
    local dev_results = {}
    for learning_rate, losses in pairs(experiment_results) do 
        train_results[#train_results+1] = {'train '.. learning_rate, losses.results_train_losses}
        dev_results[#dev_results+1] = {'dev '.. learning_rate, losses.results_dev_losses}
    end

    gnuplot.pngfigure(plotfolder..'/traintest.png')
    gnuplot.xlabel('epoch')
    gnuplot.ylabel('BCE Loss')
    gnuplot.title('Losses On Training Set')
    gnuplot.plot(unpack(train_results))
    gnuplot.plotflush()

    gnuplot.pngfigure(plotfolder..'/testest.png')
    gnuplot.xlabel('epoch')
    gnuplot.ylabel('MSE Loss')
    gnuplot.title('Losses On Development Set')    
    gnuplot.plot(unpack(dev_results))
    gnuplot.plotflush()
end

-- plot_experiment_results(common_mp.results_folder .. '/experiment_results.t7', 'plots')
-- plot_train_losses(common_mp.results_folder .. '/losses,lr=0.0005_results.t7', 'hey')


