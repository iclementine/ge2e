import paddle
import torch
import numpy as np
paddle.set_device("cpu")

__DEBUG__ = {}

class TorchModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(3, 4 * 5)
        self.l2 = torch.nn.Linear(5, 4 * 1)

    def forward(self, x, y):
        p1 = torch.tanh(self.l1(x)).reshape(-1, 5) # (5*4, 5)
        p2 = torch.tanh(self.l2(y)).reshape(-1, 1) # (5*4, 1)
        __DEBUG__["torch_p1"] = p1; p1.retain_grad()
        index = torch.repeat_interleave(torch.arange(5), 4).unsqueeze(-1)
        print("index:", index)
        p = torch.scatter(p1, 1, index, p2)
        p = p.reshape(20, -1)
        return p

class PaddleModel(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.l1 = paddle.nn.Linear(3, 4 * 5)
        self.l2 = paddle.nn.Linear(5, 4 * 1)

    def forward(self, x, y):
        p1 = paddle.tanh(self.l1(x)).reshape([-1, 5]) # (5*4, 4)
        __DEBUG__["paddle_p1"] = p1
        p1 = p1.reshape([-1])
        p2 = paddle.tanh(self.l2(y)).reshape([-1]) # (5*4, 1)
        index = paddle.arange(0, 5 * 4, dtype="int64").reshape([5, 4])
        index = index * 5 + paddle.arange(0, 5, dtype="int64").unsqueeze(-1)
        index = index.reshape([-1])
        ones = paddle.ones([5 * 4 * 5])
        zeros = paddle.zeros_like(index, dtype=ones.dtype)
        mask_p1 = paddle.scatter(ones, index, zeros)
        p = p1 * mask_p1 + (1 - mask_p1) * paddle.scatter(ones, index, p2)
        #print("index:", index.unsqueeze(-1))
        #p = paddle.scatter(p1, index, p2)
        p = p.reshape([20, -1])
        return p

model1 = TorchModel()
model2 = PaddleModel()

def convert(torch_model, paddle_model):
    paddle_model.l1.weight.set_value(torch_model.l1.weight.data.cpu().numpy().T)
    paddle_model.l2.weight.set_value(torch_model.l2.weight.data.cpu().numpy().T)
    paddle_model.l1.bias.set_value(torch_model.l1.bias.data.cpu().numpy())
    paddle_model.l2.bias.set_value(torch_model.l2.bias.data.cpu().numpy())

convert(model1, model2)

x = np.random.randn(5, 3).astype(np.float32)
y = np.random.randn(5, 5).astype(np.float32)


out1 = model1(torch.from_numpy(x.copy()), torch.from_numpy(y.copy()))
out2 = model2(paddle.to_tensor(x.copy()), paddle.to_tensor(y.copy()))

print("output: ")
print(out1.data.cpu().numpy())
print("=========")
print(out2.numpy())
print("=========")

out1.sum().backward()
out2.sum().backward()

print("grad: ")
print(model1.l1.weight.grad.data.cpu().numpy())
print("=========")
print(model2.l1.weight.grad.T)
print("=========")

print("grad of p1: ")
print(__DEBUG__["torch_p1"].grad.data.cpu().numpy())
print("=========")
print(__DEBUG__["paddle_p1"].grad)
print("=========")

