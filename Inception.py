import torch
import torch.nn as nn
from basic import BasicConv2d


class Inception(nn.Module):
    """
    GoogLeNet的Inception模块
    """
    def __init__(
            self,
            input_channels,
            outchannel1,
            inchannel2,outchannel2,
            inchannel3,outchannel3,
            outchannel4,
    ):
        """
        一个Inception模块有4个分支,会有4组输入输出通道
        各个分支通道数变化:
            分支1: input_channels-outchannel1
            分支2: input_channels-inchannel2-outchannel2
            分支3: input_channels-inchannel3-outchannel3
            分支4: input_channels-input_channels-outchannel4
        """
        super().__init__()

        self.branch1 = BasicConv2d(input_channels, outchannel1, 1)

        self.branch2 = nn.Sequential(
            BasicConv2d(input_channels, inchannel2, 1),
            BasicConv2d(inchannel2, outchannel2, 3, padding=1)
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(input_channels, inchannel3, 1),
            BasicConv2d(inchannel3, outchannel3, 5, padding=2)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3,stride=1, padding=1),
            BasicConv2d(input_channels, outchannel4, 1)
        )


    def forward(self, input):
        return torch.cat(
            tensors = [
                self.branch1(input),
                self.branch2(input),
                self.branch3(input),
                self.branch4(input)
            ],
            dim = 1
        )