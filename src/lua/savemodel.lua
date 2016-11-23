require 'nn'
require 'cunn'

net = nn.Linear(2,3)
net:cuda()

input = torch.rand(8,2)
input = input:cuda()
output = net:forward(input)

print(net:type())
print(output:type())

checkpoint = {}
checkpoint.net = net:clone():float()
print(checkpoint.net:type())

torch.save('model.t7',checkpoint)

