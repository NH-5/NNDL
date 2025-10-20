import torch
import torch.nn as nn


class FNNet(nn.Module):
    """
    一个前馈全连接网络,选用sigmoid函数作为激活函数
    size:list,元素表示神经元个数,例如[2,3,1]表示输入层2个神经元,1个隐藏层3个神经元,输出层1个神经元
    """
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.num_layers = len(size)
        self.network()

    def network(self):
        layers = []
        for i in range(self.num_layers - 1):
            layers.append(nn.Linear(self.size[i], self.size[i+1]))
            if i < self.num_layers - 2:
                layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(1,-1)
        return self.layers(x)
    

if __name__ == '__main__':
    """调试测试用"""
    input = torch.tensor([1.0,2.0,3.0],requires_grad=True)
    net = FNNet([3,2,3,1])
    output = net(input)
    print(output)
