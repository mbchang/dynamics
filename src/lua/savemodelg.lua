torch.setdefaulttensortype('torch.FloatTensor')
require 'nn'
require 'nngraph'
require 'cunn'

net = nn.Linear(2,3)
net_in = nn.Identity()()
net_out = net(net_in)
netgraph = nn.gModule({net_in},{net_out})
netgraph:cuda()

input = torch.rand(8,2)
input = input:cuda()
output = netgraph:forward(input)

print('netgraph type', netgraph:type())
print('output type', output:type())

checkpoint = {}
checkpoint.net = netgraph:clone():float()
print('checkpoint net type', checkpoint.net:type())

torch.save('model.t7',checkpoint)

