require 'torch'
require 'metaparams'
local model_utils = require 'model_utils'

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

for index, learning_rate in pairs(learning_rates) do
    print('Learning rate:', learning_rate)
    trainer:reset(learning_rate)
    optim_state = {learningRate   = learning_rate,
                         momentumDecay  = 0.1,
                         updateDecay    = 0.01}

    local train_losses = {}
    local dev_losses = {}

    local oldp, oldgp

    for i = 1, trainer.mp.max_epochs do
        if i == 3 then assert(false) end

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
end
