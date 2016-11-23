require 'torch'
require 'gnuplot'
require 'paths'
require 'utils'
torch.setdefaulttensortype('torch.FloatTensor')


local cmd = torch.CmdLine()
cmd:option('-infolder', "in", 'infolder')
cmd:option('-hid', false, 'false for training curve, true for hid state scatter plot')
cmd:text()

-- parse input params
pp = cmd:parse(arg)

-- print(pp)
-- assert(false)

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
    -- print(data)

    -- test convergence
    local val = data[{{},{2}}]
    for w =3, data:size(1) do
        local valwin = torch.exp(val[{{-w,-1}}])
        local max_val_loss, max_val_loss_idx = torch.max(valwin,1)
        local min_val_loss, min_val_loss_idx = torch.min(valwin,1)
        local val_avg_delta = (valwin[{{2,-1}}] - valwin[{{1,-2}}]):mean()
        local abs_delta = (max_val_loss-min_val_loss):sum()
        -- print(w, 'max', max_val_loss:sum(), 'maxid', max_val_loss_idx:sum(),
        --       'min', min_val_loss:sum(), 'minid', min_val_loss_idx:sum(),
        --       'avg_delta', val_avg_delta, 'abs_delta', abs_delta)
    end

    -- assert(false)
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

function plot_inference(infolder, savefile)
    if paths.filep(infolder..'/mass_infer_cf.log') then
        plot_minf(infolder..'/mass_infer_cf.log', infolder..'/mass_infer_cf.png')
    end
    if paths.filep(infolder..'/size_infer_cf.log') then
        plot_minf(infolder..'/size_infer_cf.log', infolder..'/size_infer_cf.png')
    end
    if paths.filep(infolder..'/objtype_infer_cf.log') then
        plot_minf(infolder..'/objtype_infer_cf.log', infolder..'/objtype_infer_cf.png')
    end
end

function plot_minf(logfile, savefile)

end

function plot_sinf(logfile, savefile)

end

function plot_oinf(logfile, savefile)

end

-- todo: move this to plot_results
function plot_hid_state(fname, x,y)
    -- plot scatter plot. TODO: later move this to an independent function
    gnuplot.pngfigure(mp.savedir..'/'..fname..'.png')
    gnuplot.xlabel('Euclidean Distance')
    gnuplot.ylabel('Hidden State Norm')
    gnuplot.title('Pairwise Hidden State as a Function of Distance from Focus Object')  -- TODO
    gnuplot.plot(x, y, '+')
    gnuplot.plotflush()
    print('Saved plot of hidden state to '..mp.savedir..'/'..fname..'.png')
end


plot_experiment(pp.infolder..'/experiment.log', pp.infolder..'/experiment.png')
plot_training_losses(pp.infolder..'/train.log',pp.infolder..'/train.png')

-- plot hidden state
if pp.hid then
    local fname = 'hidden_state_all_testfolders'
    local hid_info = torch.load(pp.infolder..'/'..fname)
    local all_euc_dist = torch.Tensor(hid_info.euc_dist)
    local all_euc_dist_diff = torch.Tensor(hid_info.euc_dist_diff)
    local all_effects_norm = torch.Tensor(hid_info.effects_norm)

    local neg_vel_idx = torch.squeeze(all_euc_dist_diff:lt(0):nonzero())  -- indices of all_euc_dist_diff that are negative
    local pos_vel_idx = torch.squeeze(all_euc_dist_diff:ge(0):nonzero())  -- >=0; moving away

    local neg_vel = all_euc_dist_diff:index(1,neg_vel_idx)
    local pos_vel = all_euc_dist_diff:index(1,pos_vel_idx)

    local euc_dist_neg_vel = all_euc_dist:index(1,neg_vel_idx)
    local euc_dist_pos_vel = all_euc_dist:index(1,pos_vel_idx)

    local norm_neg_vel = all_effects_norm:index(1,neg_vel_idx)
    local norm_pos_vel = all_effects_norm:index(1,pos_vel_idx)

    -- plot_hidden_state(pp.infolder..'/'..fname..'.png', all_euc_dist, all_effects_norm, pp.infolder)
    plot_hid_state(pp.infolder..'/'..fname..'_toward.png', euc_dist_neg_vel, norm_neg_vel)
    plot_hid_state(pp.infolder..'/'..fname..'_away.png', euc_dist_pos_vel, norm_pos_vel)
end