import torch
import torch.nn as nn
from Inception import Inception as inc
from basic import BasicConv2d as bc
from basic import BasicConvMaxPool as bcm


class googleNet(nn.Module):
    """
    类GoogLeNet结构的CNN部分
    结构:conv-maxpool-conv-conv-inception-[size(fc)-softmax]
    """
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            bcm(
                inch=3,
                outch=64,
                convkernel=7,
                convstride=1,
                convpadding=3,
                poolkernel=3,
                poolstride=2,
                poolpadding=1
            ),
            nn.BatchNorm2d(64),
            bc(
                input_channels=64,
                output_channels=192,
                kernel_size=1,
            ),
            bc(
                input_channels=192,
                output_channels=192,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(192),
            nn.MaxPool2d(3,2,1),
            inc(192,64,96,128,16,32,32),
            inc(256,128,128,192,32,96,64),
            nn.MaxPool2d(3,2,1),
            inc(480,192,96,208,16,48,64),
            inc(512,160,221,224,24,64,64),
            inc(512,128,128,256,24,64,64),
            inc(512,112,144,288,32,64,64),
            inc(528,256,160,320,32,128,128),
            nn.MaxPool2d(3,2,1),
            inc(832,256,160,320,32,128,128),
            inc(832,384,192,384,48,128,128),
            nn.AvgPool2d(7,1),
            nn.Dropout2d(p=0.4),
            nn.Flatten()
        )

    def forward(self, input):
        return self.model(input)

    def get_flatten_dim(self,input_size):
        input = torch.randn(1, 3, *input_size)
        out = self.forward(input)
        return out[0].numel()   # 统计所有元素个数


if __name__ == '__main__':
    """调试用"""
    net = googleNet()
    flatten = net.get_flatten_dim((224,224))
    print(flatten)

