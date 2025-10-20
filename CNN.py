import torch
import torch.nn as nn


class CNNet(nn.Module):
    """
    仿LeNet-5结构
    conv->pool->conv->pool
    采用ReLU作为激活函数
    """
    
    def __init__(self, input_channels=3):
        super(CNNet, self).__init__()
        
        self.input_channels = input_channels
        self.network()

    def network(self):
        module = nn.ModuleList()
        module.append(
            nn.Conv2d(
                in_channels=self.input_channels,
                out_channels=6,
                kernel_size=5,
                stride=1
            )
        )
        module.append(nn.ReLU())
        module.append(
            nn.AvgPool2d(
                kernel_size=2,
                stride=2
            )
        )
        module.append(nn.ReLU())
        module.append(
            nn.Conv2d(
                in_channels=6,
                out_channels=16,
                kernel_size=5,
                stride=1
            )
        )
        module.append(nn.ReLU())
        module.append(
            nn.AvgPool2d(
                kernel_size=2,
                stride=2
            )
        )
        module.append(nn.ReLU())
        self.module = nn.Sequential(*module)

    def forward(self, x):
        return self.network(x)
        
        