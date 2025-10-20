import torch
import torch.nn as nn


class CNNet(nn.Module):
    """
    仿LeNet-5结构
    conv->pool->conv->pool
    采用ReLU作为激活函数
    """
    
    def __init__(self, input_channels=3, input_size = (360, 640), is_flatten = False):
        super(CNNet, self).__init__()
        
        self.input_channels = input_channels
        self.input_size = input_size
        self.is_flatten = is_flatten
        self.network()

    def network(self):
        model = nn.ModuleList()
        model.append(
            nn.Conv2d(
                in_channels=self.input_channels,
                out_channels=6,
                kernel_size=5,
                stride=1
            )
        )
        model.append(nn.ReLU())
        model.append(
            nn.AvgPool2d(
                kernel_size=2,
                stride=2
            )
        )
        model.append(nn.ReLU())
        model.append(
            nn.Conv2d(
                in_channels=6,
                out_channels=16,
                kernel_size=5,
                stride=1
            )
        )
        model.append(nn.ReLU())
        model.append(
            nn.AvgPool2d(
                kernel_size=2,
                stride=2
            )
        )
        model.append(nn.ReLU())
        if self.is_flatten:
            model.append(nn.Flatten(start_dim=1))
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
    

    def get_output_shape(self):
        """
        input_shape: (C, H, W) 例如 (3, 360, 640)
        """
        # 增加 batch 维度 => (1, C, H, W)
        dummy = torch.randn(1, self.input_channels, *self.input_size)
        out = self.forward(dummy)
        return out.shape   # 返回完整shape：(batch, C, H, W)
    

    def get_flatten_dim(self):
        dummy = torch.randn(1, self.input_channels, *self.input_size)
        out = self.forward(dummy)
        return out[0].numel()   # 统计所有元素个数

        
if __name__ == '__main__':
    """调试用"""
    input = torch.rand(2,2)
    net = CNNet(input_channels=1, input_size=(100,100), is_flatten=True)
    output_shape = net.get_output_shape()
    output_dim = net.get_flatten_dim()
    print(output_shape)
    print(output_dim)
