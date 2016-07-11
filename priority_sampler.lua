
local priority_sampler = {}
priority_sampler.__index = priority_sampler

function priority_sampler.create(num_batches)
    local self = {}
    setmetatable(self, priority_sampler)
    self.batch_weights = torch.zeros(num_batches)
    self.alpha = 0.1 -- corresponds to when we do random sampling
    self.epc_num = 1
    self.table_is_full = false
    self.num_batches = num_batches

    collectgarbage()
    return self
end

function priority_sampler:update_batch_weight(batch_id, weight)
    self.batch_weights[batch_id] = weight
    self.table_is_full = self.batch_weights:min() > 0
end

function normalize(tensor)
    local out = tensor:clone()
    if out:min() <= 0 then
        local m, am = torch.min(out,1)
        print('min',m)
        print('amin',am)
    end
    assert(out:min() > 0)
    local result = out/out:sum()
    return result
end

-- you should call this method after about 2 epochs or something
function priority_sampler:sample(pow)
    local batch_id
    if math.random(100)/100.0 < self.alpha then
        batch_id = math.random(self.num_batches)
    else
        if not pow then pow = 1 end
        local sharpened = torch.pow(self.batch_weights, pow)
        local normalized = normalize(sharpened)
        batch_id = torch.multinomial(normalized,1,true):sum()
    end
    return batch_id
end

function priority_sampler:get_hardest_batch()
    local max, argmax = torch.max(self.batch_weights,1)
    assert(max:dim() == 1 and argmax:dim() == 1)
    return {max:sum(), argmax:sum()}
end

-- can do some sort of annealing
function priority_sampler:set_alpha(newvalue)
    self.alpha = newvalue
end

function priority_sampler:set_epcnum(new_epcnum)
    self.epc_num = new_epcnum
end


return priority_sampler
