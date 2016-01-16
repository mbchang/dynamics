require 'torch'
local model_utils = require 'model_utils'

require 'nn'
-- require 'randomkit'
require 'optim'
require 'image'
-- require 'dataset-mnist'
-- require 'cutorch'
require 'xlua'
require 'Base'
require 'sys'
require 'pl'
-- local T = require 'pl.tablex'


-- TODO: Have a shell script for openmind and for pc

-- missing params: seed, experiment_string
-- also booleans are strings somehow
-- you can
--------------------------- Init ------------------------------
require 'logging_utils'


params = lapp[[
   -d,--savedir       (default "logs")      	subdirectory to save logs
   -m,--model         (default "lstm")   		type of model tor train: lstm |
   -p,--plot                                	plot while training
   -c,--server		  (default "pc")			pc=personal | op = openmind
   -s,--shuffle  	  (default false)
   -r,--lr            (default 0.0005)      	learning rate
   -i,--max_epochs    (default 50)           	maximum nb of iterations per batch, for LBFGS
   --batch_size       (default 1)           	1 pc, 100 op
   --rnn_dim          (default 100)
   --seq_length       (default 10)				10 pc, 20 op
   --layers           (default 4)
   --rand_init_wts    (default false)
   --max_grad_norm    (default 10)
   --save_every		  (default 50)
   --print_every	  (default 10)
]]



if params.rand_init_wts then torch.manualSeed(123) end
if params.server == 'pc' then
	params.winsize = 10
	params.dataset_folder = 'hey'
	params.batch_size = 1
	params.seq_length = 10
	params.num_threads = 1
	params.cuda = false
	params.cunn = false
else
	params.winsize = 20
	params.dataset_folder = '/om/user/mbchang/physics-data/dataset_files'
	params.batch_size = 100
	params.seq_length = 20
	params.num_threads = 4
	params.cuda = true
	params.cunn = true
end


params.input_dim = 8.0*params.winsize/2
params.out_dim = 8.0*params.winsize/2
params.results_folder = create_experiment_string({'batch_size', 'seq_length', 'layers', 'rnn_dim', 'max_epochs'}, params) .. 'new_test'


common_mp = tablex.deepcopy(params)
train_mp = tablex.deepcopy(params)
test_mp = tablex.deepcopy(params)


if common_mp.cuda then require 'cutorch' end
if common_mp.cunn then require 'cunn' end

-- threads
-- torch.setnumthreads(common_mp.num_threads)
-- print('<torch> set nb of threads to ' .. torch.getnumthreads())

local Trainer = require 'train'
local Tester = require 'test'

local trainer = Trainer.create('trainset', train_mp)  -- need to specify learning rate here
local trainer_tester = Tester.create('trainset', test_mp)
local tester = Tester.create('testset', test_mp)

local learning_rates = {5e-4, 5e-5, 5e-6}

local experiment_results = common_mp.results_folder .. '/experiment_results.t7'

if not common_mp.rand_init_wts then torch.manualSeed(123) end
torch.manualSeed(123)

local all_results = {}

-- for index, learning_rate in pairs(learning_rates) do
local learning_rate = params.lr
print('Learning rate:', learning_rate)
trainer:reset(learning_rate)
local train_losses = {}
local dev_losses = {}

local oldp, oldgp

for i = 1, trainer.mp.max_epochs do

    -- Train
    -- this train_loss is the final loss after one epoch. We expect to see this go down as epochs increase
    local model
    local train_loss
    if train_mp.curriculum then
        _, model = trainer:curriculum_train(1, i)  -- trainer.train_loader.num_batches  TODO: don't make it avg train loss in curriculum!
    else
        train_loss, model = trainer:train(trainer.train_loader.num_batches, i)  -- trainer.train_loader.num_batches
    end
    -- local _, model = trainer:curriculum_train(1, i)  -- trainer.train_loader.num_batches

    -- Get the training loss
    -- local train_loss = trainer_tester:test(model, p, trainer_tester.test_loader.num_batches)  -- tester.test_loader.nbatches  -- creating new copy of model when I load into Tester!
    -- local train_loss = trainer_tester:test(model, trainer_tester.test_loader.num_batches)  -- tester.test_loader.nbatches  -- creating new copy of model when I load into Tester!
    print('train loss\t', train_loss)

    -- Test
    -- this train_loss is the final loss after one epoch. We expect to see this go in a parabola as epochs increase
    -- local dev_loss = tester:test(model, p, tester.test_loader.num_batches)  -- tester.test_loader.nbatches  -- creating new copy of model when I load into Tester!
    local dev_loss = tester:test(model, tester.test_loader.num_batches)  -- tester.test_loader.nbatches  -- creating new copy of model when I load into Tester!

    -- Record loss
    train_losses[#train_losses+1] = train_loss
    dev_losses[#dev_losses+1] = dev_loss
    print('avg dev loss\t', dev_loss)
    -- When we save results, for this learning rate, this is the curve of train_loss and dev_loss as we go through epochs

    print('train_losses:', train_losses)
    print('dev_losses:', dev_losses)

    -- Can save here actually, so that you constantly save stuff. Basically you rewrite all_results[learning_rate] each time
    all_results[learning_rate] = {results_train_losses  = torch.Tensor(train_losses),
                                  results_dev_losses    = torch.Tensor(dev_losses)}
    print('all_results', all_results)
    torch.save(experiment_results, all_results)

    if common_mp.cuda then cutorch.synchronize() end
    collectgarbage()
end
-- end
