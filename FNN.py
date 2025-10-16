import torch
import torch.nn as nn
import torch.nn.functional as F


class FNNet(nn.Module):
    """
    全连接神经网络模块
    
    参数:
        input_size: 输入特征的维度（若与CNN链接则来自CNN的展平输出）
        fnn_structure: FNN各层的神经元个数列表
            例如: [512, 256] 表示两个隐藏层，分别有512和256个神经元
        num_classes: 分类类别数，默认10
        dropout_rate: Dropout比例，默认0.5
    """
    
    def __init__(self, input_size, fnn_structure, num_classes=10, dropout_rate=0.5):
        super(FNNet, self).__init__()
        
        self.input_size = input_size
        self.fnn_structure = fnn_structure
        self.num_classes = num_classes
        
        # 构建全连接层
        self.fc_layers = nn.ModuleList()
        
        # 第一层：从输入到第一个隐藏层
        if len(fnn_structure) > 0:
            self.fc_layers.append(nn.Linear(input_size, fnn_structure[0]))
            
            # 中间的隐藏层
            for i in range(len(fnn_structure) - 1):
                self.fc_layers.append(nn.Linear(fnn_structure[i], fnn_structure[i+1]))
            
            # 最后一层：输出层
            self.fc_layers.append(nn.Linear(fnn_structure[-1], num_classes))
        else:
            # 如果没有隐藏层，直接连接到输出
            self.fc_layers.append(nn.Linear(input_size, num_classes))
        
        # Dropout层，防止过拟合
        self.dropout = nn.Dropout(dropout_rate)
        
    
    def forward(self, x):
        # 通过全连接层（除了最后一层）
        for i, fc in enumerate(self.fc_layers[:-1]):
            x = fc(x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # 最后一层（输出层）
        x = self.fc_layers[-1](x)
        
        return x


# 测试代码
if __name__ == '__main__':
    # 假设CNN输出的特征维度为4096
    input_size = 4096
    fnn_structure = [512, 256]
    
    fnn = FNNet(input_size=input_size, fnn_structure=fnn_structure, num_classes=10)
    
    print(f"FNN输入维度: {input_size}")
    print(f"FNN隐藏层结构: {fnn_structure}")
    print(f"输出类别数: {fnn.num_classes}")
    
    # 测试前向传播
    dummy_input = torch.randn(4, input_size)  # batch_size=4
    output = fnn(dummy_input)
    print(f"\n输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape}")