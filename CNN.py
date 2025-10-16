import torch
import torch.nn as nn


class CNNet(nn.Module):
    """
    卷积神经网络模块
    
    参数:
        conv_layers: (Conv2d + MaxPool2d) 组合的数量
        input_channels: 输入图像的通道数 (默认3, RGB图像)
        input_size: 输入图像的尺寸 (height, width)，默认(360, 640)
    """
    
    def __init__(self, conv_layers, input_channels=3, input_size=(360, 640)):
        super(CNNet, self).__init__()
        
        self.conv_layers = conv_layers
        self.input_size = input_size
        
        # 构建卷积层
        self.conv_blocks = nn.ModuleList()
        in_channels = input_channels
        out_channels = 32  # 初始卷积核数量
        
        for i in range(self.conv_layers):
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.conv_blocks.append(conv_block)
            in_channels = out_channels
            out_channels = min(out_channels * 2, 512)  # 逐层增加通道数，最多512
        
        # 计算卷积层输出后的特征图尺寸
        h, w = input_size
        for _ in range(self.conv_layers):
            h = h // 2
            w = w // 2
        
        # 展平后的特征数量
        self.flatten_size = in_channels * h * w
        self.output_channels = in_channels
        
    
    def forward(self, x):
        # 通过所有卷积块
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        return x
    
    
    def get_output_size(self):
        """返回卷积层输出的展平尺寸"""
        return self.flatten_size


# 测试代码
if __name__ == '__main__':
    # 创建CNN模块
    cnn = CNNet(conv_layers=3, input_channels=3, input_size=(360, 640))
    
    print(f"CNN卷积层数: {cnn.conv_layers}")
    print(f"输出特征维度: {cnn.get_output_size()}")
    
    # 测试前向传播
    dummy_input = torch.randn(4, 3, 360, 640)  # batch_size=4
    output = cnn(dummy_input)
    print(f"\n输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape}")