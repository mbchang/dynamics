
local priority_sampler = {}
priority_sampler.__index = dataloader

function priority_sampler.create(num_train_exs)
    local self = {}
    setmetatable(self, priority_sampler)

    self.batch_weights = torch.zeros(num_train_exs)
    self.alpha = 0.9 -- corresponds to when we do random sampling

    collectgarbage()
    return self
end

function priority_sampler:update_batch_weight(batch_id, weight)
    self.batch_weights[batch_id] = weight
end

function priority_sampler:normalize()
    -- make sure they are nonzero, it is very hard to get zero loss
    assert(self.batch_weights:min() > 0)
    self.batch_weights = self.batch_weights/self.batch_weights:sum()
end

function priority_sampler:sample()
    self:normalize()  -- make sure values are between 0 and 1
    local batch_id
    if math.random(100)/100.0 < self.alpha then
        batch_id = math.random(num_train_ex)
    else
        batch_id = torch.multinomial(self.batch_weights,1,true)
    end
end

-- can do some sort of annealing
function priority_sampler:set_alpha(newvalue)
    self.alpha = newvalue
end


return priority_sampler
