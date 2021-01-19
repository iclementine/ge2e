import numpy as np
import torch
import paddle 
paddle.set_device("cpu")

import model
import torch_model

model1 = torch_model.SpeakerEncoder(mel_dim=40,
                                    num_layers=3,
                                    hidden_size=32,
                                    output_size=128)
optim1 = torch.optim.SGD(model1.parameters(), lr=0.001)
model2 = model.SpeakerEncoder(n_mel=40,
                              num_layers=3,
                              hidden_size=32,
                              output_size=128)
optim2 = paddle.optimizer.SGD(0.001, parameters=model2.parameters())

def convert_model(torch_model, paddle_model):
    paddle_model.similarity_weight.set_value(torch_model.similarity_weight.data.cpu().numpy())
    paddle_model.similarity_bias.set_value(torch_model.similarity_bias.data.cpu().numpy())
    for k, v in torch_model.lstm.named_parameters():
        paddle_model.lstm._parameters[k].set_value(v.data.cpu().numpy())
    paddle_model.linear.weight.set_value(torch_model.linear.weight.data.cpu().numpy().T)
    paddle_model.linear.bias.set_value(torch_model.linear.bias.data.cpu().numpy())

convert_model(model1, model2)

utterance = np.random.randn(5*4, 64, 40).astype(np.float32) # B,T, C
embed1 = model1(torch.from_numpy(utterance))
embed2 = model2(paddle.to_tensor(utterance))

# backward embed, it has been tested
if False:
    print("embed: ")
    print(embed1.data.cpu().numpy())
    print("=============")
    print(embed2.numpy())
    print("=============")

    embed1.sum().backward()
    embed2.sum().backward()

    print("lstm grads: ")
    print(model1.lstm.weight_hh_l0.grad)
    print("=============")
    print(model2.lstm.weight_hh_l0.grad)
    print("=============")
    exit(0)

# 正向反向都验证过了, 直接 embed.sum().backward() 然后查看梯度就可以了。
M1, p1, q1 = model1.similarity_matrix(embed1.reshape(5, 4, -1))
M2, p2, q2 = model2.similarity_matrix(embed2.reshape([5, 4, -1]))

# backward sim matrix
if False:
    print("similarity matrix: ")
    print(M1.data.cpu().numpy())
    print("=============")
    print(M2.numpy())
    print("=============")
    M1.sum().backward()
    M2.sum().backward()

    import pdb; pdb.set_trace()
    print("linear weight grad: ")
    print(model1.linear.weight.grad.data.cpu().numpy())
    print("=============")
    print(model2.linear.weight.grad.T)
    print("=============")
    import pdb; pdb.set_trace()
    exit(0)

loss1, eer1 = model1.loss(embed1.reshape(5, 4, -1))
loss2, eer2 = model2.loss(embed2.reshape([5, 4, -1]))


if True:
    # TODO: 继续对齐 Loss backward 的梯度
    print("loss1: ", float(loss1))
    print("loss2: ", float(loss2))
    loss1.backward()
    loss2.backward()

    print("model1 grad:", float(model1.similarity_bias.grad))
    print("model2 grad:", float(model2.similarity_bias.grad))

    print("linear weight grad: ")
    print(model1.lstm.weight_hh_l0.grad.data.cpu().numpy())
    print("=============")
    print(model2.lstm.weight_hh_l0.grad)
    print("=============")
    import pdb; pdb.set_trace()
    exit(0)




optim1.step()
optim2.step()

optim1.zero_grad()
optim2.clear_grad()

# for k, v in model1.named_parameters():
#     print(k, v.shape)

# print("======")
# for k, v in model2.state_dict().items():
#     print(k, v.shape)

