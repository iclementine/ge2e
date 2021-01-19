import paddle
import torch
import numpy as np

x = np.random.randn(32, 1, 8).astype(np.float32)
y = np.random.randn(32, 8, 1).astype(np.float32)

x1 = torch.from_numpy(x.copy()).requires_grad_()
y1 = torch.from_numpy(y.copy()).requires_grad_()

torch.bmm(torch.relu(x1), y1).sum().backward()
print("torch bmm: ")
print(x1.grad.data.cpu().numpy().squeeze())
print("===========")
print(y1.grad.data.cpu().numpy().squeeze())
print("===========")


x2 = paddle.to_tensor(x.copy())
x2.stop_gradient = False
y2 = paddle.to_tensor(y.copy())
y2.stop_gradient = False

paddle.bmm(paddle.nn.functional.relu(x2), y2).sum().backward()
print("paddle bmm: ")
print(x2.grad.squeeze())
print("===========")
print(y2.grad.squeeze())
print("===========")


