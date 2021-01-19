import paddle
from paddle import nn
from paddle.nn.clip import ClipGradByGlobalNorm
paddle.set_device("cpu")

x = paddle.randn([4, 6])
net = nn.Linear(6, 3)
y = net(x)
y.sum().backward()

print("grad in numpy:\n", net.weight.grad)


p = net.weight._grad_ivar()
with paddle.no_grad():
    p[...] = p * 0.01

print("grad in numpy:\n", net.weight.grad)