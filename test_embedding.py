import torch 
import paddle
import numpy as np
np.random.seed(1)
paddle.set_device("gpu:0")

net1 = torch.nn.Embedding(20, 6, padding_idx=0)
net2 = paddle.nn.Embedding(20, 6, padding_idx=0)
weight = np.random.uniform(-0.5, 0.5, (20, 6)).astype(np.float32)
net2.weight.set_value(weight)
net1.weight.data[:] = torch.from_numpy(weight)
optim1 = torch.optim.SGD(net1.parameters(), 0.1)
optim2 = paddle.optimizer.SGD(0.1, parameters=net2.parameters())

ids = np.array([0, 0, 0, 0, 0, 1, 2, 12])

embed1 = net1(torch.from_numpy(ids.copy()))
embed2 = net2(paddle.to_tensor(ids.copy()))

print("Forward")
print(embed1.data.cpu().numpy()[3])
print("==========")
print(embed2.numpy()[3])
print("==========")

print("Weights")
print(net1.weight.data.cpu().numpy())
print("==========")
print(net2.weight.numpy())
print("==========")

torch.tanh(embed1).sum().backward()
paddle.tanh(embed2).sum().backward()

print("Grads:")
print(net1.weight.grad.data.cpu().numpy())
print("==========")
np.set_printoptions(precision=8)
print(net2.weight.grad)

optim1.step()
optim2.step()

print("Updated Weights")
print(net1.weight.data.cpu().numpy())
print("==========")
print(net2.weight.numpy())
print("==========")

import pdb; pdb.set_trace()
print("Done")