import torch
import torch.nn as nn


class BasicConv2d(nn.Module):
    """
    单独的卷积模块"conv-ReLU"
    """
    def __init__(
            self,
            input_channels : int,
            output_channels : int,
            kernel_size : int,
            stride : int = 1,
            padding : int = 0
        ):
        super().__init__()

        self.conv2d = nn.Conv2d(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.relu = nn.ReLU()

        self.model = nn.Sequential(
            self.conv2d,
            self.relu
        )

    def forward(self, input):
        return self.model(input)
        

class BasicConvMaxPool(nn.Module):
    """
    单独的卷积池化模块"conv-ReLU-maxpool"
    """
    def __init__(
            self,
            inch : int,
            outch : int,
            convkernel : int,
            convstride : int = 1,
            convpadding : int = 0,
            poolkernel : int = 2,
            poolpadding : int = 0,
            poolstride : int = 2
    ):
        super().__init__()

        self.conv2d = nn.Conv2d(
            in_channels=inch,
            out_channels=outch,
            kernel_size=convkernel,
            stride=convstride,
            padding=convpadding
        )
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(
            kernel_size=poolkernel,
            stride=poolstride,
            padding=poolpadding
        )

        self.model = nn.Sequential(
            self.conv2d,
            self.relu,
            self.maxpool
        )


    def forward(self, input):
        return self.model(input)
    

class BasicConvAvgPool(nn.Module):
    """
    单独的卷积池化模块"conv-ReLU-avgpool"
    """
    def __init__(
            self,
            inch : int,
            outch : int,
            convkernel : int,
            convstride : int = 1,
            convpadding : int = 0,
            poolkernel : int = 2,
            poolpadding : int = 0,
            poolstride : int = 2
    ):
        super().__init__()

        self.conv2d = nn.Conv2d(
            in_channels=inch,
            out_channels=outch,
            kernel_size=convkernel,
            stride=convstride,
            padding=convpadding
        )
        self.relu = nn.ReLU()
        self.maxpool = nn.AvgPool2d(
            kernel_size=poolkernel,
            stride=poolstride,
            padding=poolpadding
        )

        self.model = nn.Sequential(
            self.conv2d,
            self.relu,
            self.maxpool
        )


    def forward(self, input):
        return self.model(input)
    

if __name__ == '__main__':
    """调试用"""
    device = torch.device('cuda')
    data = torch.randn(3,5,5,device=device)
    net = BasicConvAvgPool(3,4,3)
    net = net.to(device)
    output = net(data)
    print(output)