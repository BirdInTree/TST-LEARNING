import torch
import torch.nn as nn

linear = nn.Linear(10,1)
conv2d = nn.Conv2d(1,2,(3,3))

torch.manual_seed(1)
print(linear.weight.data)
print(f"初始化前的均值{linear.weight.data.mean()},标准差{linear.weight.data.std()}")

#initial
torch.nn.init.normal_(linear.weight.data, mean=0, std=0.01)
print(linear.weight.data)
print(f"初始化后的均值{linear.weight.data.mean()},标准差{linear.weight.data.std()}")

#constant init
print(conv2d.weight.data)
torch.nn.init.constant_(conv2d.weight.data, 0.1)
print(conv2d.weight.data)

# 对conv进行kaiming初始化
torch.nn.init.kaiming_normal_(conv2d.weight.data,mode='fan_in')
print(conv2d.weight.data)