require 'nn'
torch.setdefaulttensortype('torch.FloatTensor')

cuda = false
if cuda then
    require 'cunn'
    require 'cutorch'
end

a = nn.Linear(50,50)

local timer = torch.Timer()

for i = 1, 1000 do
    local b = torch.rand(50,50)
    if cuda then 
        b = b:cuda() 
        a:cuda()
    end

    local o = a:forward(b)
    a:backward(b,o)
end
print('Time elapsed for 1,000,000 sin: ' .. timer:time().real .. ' seconds')

