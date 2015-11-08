require 'torch'
require 'metaparams'
local model_utils = require 'model_utils'

if common_mp.cuda then
    require 'cutorch'
    require 'cunn'
end

-- threads
torch.setnumthreads(4)
print('<torch> set nb of threads to ' .. torch.getnumthreads())

local Trainer = require 'pe_train'
local Tester = require 'pe_test'

local trainer = Trainer.create('data_small', train_mp, train_mp_ignore)  -- need to specify learning rate here
local tester = Tester.create('data_small', test_mp)

local learning_rates = {5e-4, 5e-5, 5e-6}
local experiment_results = common_mp.results_folder .. '/experiment_results.t7'

if common_mp.rand_init_wts then torch.manualSeed(123) end

local all_results = {}

for index, learning_rate in pairs(learning_rates) do 
    print('Learning rate:', learning_rate)
    trainer:reset(learning_rate)
    local train_losses = {}
    local dev_losses = {}

    for i = 1, trainer.mp.max_epochs do
        -- Train
        -- this train_loss is the final loss after one epoch. We expect to see this go down as epochs increase
        local train_loss, model = trainer:train(20, i)  -- trainer.train_loader.nbatches  
        local p, gp = model_utils.combine_all_parameters(unpack(model)) -- model:getParameters()
        local paramNorm, gpNorm = p:norm(), gp:norm()

        -- Test
        -- this train_loss is the final loss after one epoch. We expect to see this go in a parabola as epochs increase
        local dev_loss = tester:test(model, 10)  -- tester.test_loader.nbatches  -- creating new copy of model when I load into Tester!
        local p, gp = model_utils.combine_all_parameters(unpack(model)) -- model:getParameters()
        assert(p:norm() == paramNorm)
        assert(gp:norm() == gpNorm)

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

-- Save plots


