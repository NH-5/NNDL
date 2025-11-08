import torch
import torch.nn as nn
from basic import BasicConvAvgPool


class leNet(nn.Module):
    """
    LeNet-5结构
    conv->pool->conv->pool
    采用ReLU作为激活函数
    """
    
    def __init__(self, input_channels=3, input_size = (360, 640), is_flatten = False):
        super().__init__()
        
        self.input_channels = input_channels
        self.input_size = input_size
        self.is_flatten = is_flatten
        self.network()

    def network(self):
        model = nn.Sequential(
            BasicConvAvgPool(self.input_channels, 6, 5),
            nn.ReLU(),
            BasicConvAvgPool(6,16,5),
            nn.ReLU()
        )
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
    net = leNet(input_channels=1, input_size=(100,100), is_flatten=True)
    output_shape = net.get_output_shape()
    output_dim = net.get_flatten_dim()
    print(output_shape)
    print(output_dim)
