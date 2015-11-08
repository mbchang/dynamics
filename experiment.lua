require 'torch'
require 'metaparams'

if common_mp.cuda then
    require 'cutorch'
    require 'cunn'
end

-- threads
torch.setnumthreads(4)
print('<torch> set nb of threads to ' .. torch.getnumthreads())

local Trainer = require 'pe_train'
local Tester = require 'pe_test'

local trainer = Trainer.create('train', train_mp, train_mp_ignore)  -- need to specify learning rate here
local tester = Tester.create('dev', test_mp)

local learning_rates = {5e-4, 5e-5, 5e-6}
local experiment_results = common_mp.results_folder .. '/experiment_results.t7'

if common_mp.rand_init_wts then torch.manualSeed(123) end

for index, learning_rate in pairs(learning_rates) do 
    print('Learning rate:', learning_rate)
    trainer:reset(learning_rate)
    local train_losses = {}






    for i = 1, trainer.mp.max_epochs do
        local train_loss, modelfile = trainer:train(200, i)  -- trainer.train_loader.nbatches



        -- TODO: trainer:train should outpu train_loss, self.protos; and tester should initialize ITS self.protos to the protos that train gives, this is NOT copy!

        p, gp = model:getParameters()
        paramNorm = p:norm()
        gpNorm = gp:norm()

        local dev_losses = {}
        local dev_loss = tester:test(modelfile, 10)  -- tester.test_loader.nbatches  -- creating new copy of model when I load into Tester!
        
        p, gp = model:getParameters()
        assert(p:norm() == paramNorm)
        assert(gp:norm() == gpNorm)

        train_losses[#train_losses+1] = train_loss
        dev_losses[#dev_losses+1] = dev_loss
        print('avg dev loss\t', dev_loss)

        local all_results = {results_learning_rates = torch.Tensor(learning_rates),
                        results_train_losses = torch.Tensor(train_losses),
                        results_dev_losses = torch.Tensor(dev_losses)}
        torch.save(experiment_results, all_results)

        if common_mp.cuda then cutorch.synchronize() end
        collectgarbage()
    end






end

