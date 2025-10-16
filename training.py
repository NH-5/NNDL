import torch
import torch.nn as nn
import torch.optim as optim
from CNN import CNNet
from FNN import FNNet
from dataLoader import PNGLoader
from notify import bark_send as bs
from logs import write_logs as wl


class GameClassifier(nn.Module):
    """
    游戏分类器：结合CNN和FNN
    """
    def __init__(self, size, fnn_structure, input_channels=3, input_size=(360, 640), num_classes=10):
        super(GameClassifier, self).__init__()
        
        # 创建CNN模块
        self.cnn = CNNet(conv_layers=size[0], input_channels=input_channels, input_size=input_size)
        
        # 获取CNN输出维度
        cnn_output_size = self.cnn.get_output_size()
        
        # 创建FNN模块
        self.fnn = FNNet(input_size=cnn_output_size, fnn_structure=fnn_structure, num_classes=num_classes)
        
    
    def forward(self, x):
        x = self.cnn(x)
        x = self.fnn(x)
        return x


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    训练一个epoch
    返回: (平均loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # 如果labels是one-hot编码，转换为类别索引
        if labels.dim() > 1:
            labels = torch.argmax(labels, dim=1)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """
    验证模型
    返回: (平均loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 如果labels是one-hot编码，转换为类别索引
            if labels.dim() > 1:
                labels = torch.argmax(labels, dim=1)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


def train_model(model, train_loader, val_loader, num_epochs, learning_rate, device):
    """
    完整的训练流程
    返回: history列表，每个元素为 [epoch, train_loss, train_acc, val_loss, val_acc]
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    history = []
    
    for epoch in range(num_epochs):
        # 训练
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # 验证
        if val_loader is not None:
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            history.append([epoch + 1, train_loss, train_acc, val_loss, val_acc])
            print(f"Epoch {epoch+1} Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        else:
            history.append([epoch + 1, train_loss, train_acc, None, None])
            print(f"Epoch {epoch+1} Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    
    return history


if __name__ == '__main__':
    # ===== 配置参数 =====
    size = [1, 2]  # [卷积层数, FNN层数]
    fnn_structure = [64, 64]  # FNN各层神经元数
    
    batch_size = 32
    num_epochs = 30
    learning_rate = 0.001
    
    # 检测设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ===== 加载数据 =====
    train_dataset, train_loader = PNGLoader('data/gameplay-images/train_data', 
                                            batch_size=batch_size, shuffle=True)
    
    # 可选：加载验证集
    try:
        val_dataset, val_loader = PNGLoader('data/gameplay-images/test_data', 
                                            batch_size=batch_size, shuffle=False)
    except:
        val_loader = None
    
    # ===== 创建模型 =====
    model = GameClassifier(
        size=size,
        fnn_structure=fnn_structure,
        input_channels=3,
        input_size=(360, 640),
        num_classes=10
    ).to(device)
    
    # ===== 训练模型 =====
    history = train_model(model, train_loader, val_loader, num_epochs, learning_rate, device)
    loss = []
    accuracy = []
    # history格式: [[epoch, train_loss, train_acc, val_loss, val_acc], ...]
    # 你可以用这个列表记录日志
    for epoch in range(num_epochs):
        loss.append(history[epoch][1])
        accuracy.append(history[epoch][4])
    # ===== 保存模型 =====
    torch.save(model.state_dict(), 'game_classifier.pth')

    # 发消息
    bs('训练结束', f'Epoch {history[-1][0]+1} Train Loss: {history[-1][1]:.4f}, Train Acc: {history[-1][2]:.4f} | Val Loss: {history[-1][3]:.4f}, Val Acc: {history[-1][4]:.4f}')
    wl(loss, accuracy, num_epochs)

