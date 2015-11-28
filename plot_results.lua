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

    local skipped_losses = {}
    local x = torch.range(1, train_losses:size(1), 100)
    for i=1,train_losses:size(1),100 do
        skipped_losses[#skipped_losses+1] = train_losses[i]
    end
    skipped_losses = torch.Tensor(skipped_losses)

    gnuplot.pngfigure(plotfolder..'/logskippedlosses.png')
    gnuplot.xlabel('batch')
    gnuplot.ylabel('Log MSE Loss')
    gnuplot.title('Losses On Training Set (Learning Rate = 0.0005)')  -- change
    gnuplot.plot({x, skipped_losses})
    gnuplot.plotflush()
end

function plot_experiment_results(experiment_results_file, plotfolder)
    if not paths.dirp(plotfolder) then 
        paths.mkdir(plotfolder)
    end

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

    gnuplot.pngfigure(plotfolder..'/traintest.png')
    gnuplot.xlabel('epoch')
    gnuplot.ylabel('Log MSE Loss')
    gnuplot.title('Losses On Training Set')
    gnuplot.plot(unpack(train_results))
    gnuplot.plotflush()

    gnuplot.pngfigure(plotfolder..'/testest.png')
    gnuplot.xlabel('epoch')
    gnuplot.ylabel('Log MSE Loss')
    gnuplot.title('Losses On Development Set')    
    gnuplot.plot(unpack(dev_results))
    gnuplot.plotflush()

    gnuplot.pngfigure(plotfolder..'/alltest.png')
    gnuplot.xlabel('epoch')
    gnuplot.ylabel('Log MSE Loss')
    gnuplot.title('Losses')    
    gnuplot.plot(unpack(all_results))
    gnuplot.plotflush()
end

-- assert(false)
common_mp.results_folder = 'rand_order_results_batch_size=100_seq_length=10_layers=4_rnn_dim=100'
print(common_mp.results_folder)

plot_experiment_results(common_mp.results_folder .. '/experiment_results.t7', 'plots')
-- plot_train_losses(common_mp.results_folder .. '/losses,lr=0.0005_results.t7', 'plots')


