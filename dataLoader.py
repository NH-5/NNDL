import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import pandas as pd

class CSVDataSet(Dataset):
    """
    自定义数据集类，用于读取 CSV 文件并转换为 Tensor.
    对于所使用的Mnist数据格式如下:
        第 0 列:标签(0~9)
        第 1~784 列:像素值(0~255)
    """
    def __init__(self, file_path):
        df = pd.read_csv(file_path)

        # 标签为 long 类型，方便后续处理
        self.labels = torch.tensor(df.iloc[:, 0].values, dtype=torch.long)
        # 像素归一化到 0~1
        self.features = torch.tensor(df.iloc[:, 1:].values, dtype=torch.float32) / 255.0

    def __len__(self):
        # 返回数据集大小
        return len(self.labels)
    
    def __getitem__(self, index):
        # 返回一个样本 (features, label)
        return self.features[index], self.labels[index]
    

def CSVLoader(path, batch_size=64, shuffle=True):
        """
        加载数据集并返回两个对象：
        1. 原始 dataset(可以用于访问完整数据或调试)
        2. DataLoader(用于 mini-batch 训练)
        
        参数：
            batch_size: 每个 batch 的大小
            shuffle: 是否在每个 epoch 开始前打乱数据
        """
        dataset = CSVDataSet(path)

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return dataset, dataloader
    

def PNGLoader(trainpath, testpath, batch_size=64, shuffle=False):
    transform = transforms.Compose(
        transforms.ToTensor()
    )
    train_dataset = ImageFolder(root=trainpath, transform=transform)
    test_dataset = ImageFolder(root=testpath, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
    return train_loader, test_loader

