import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class MnistDataSet(Dataset):
    """
    自定义数据集类，用于读取 MNIST CSV 文件并转换为 Tensor。
    数据格式：
        第 0 列：标签（0~9）
        第 1~784 列：像素值（0~255）
    """
    def __init__(self, file_path):
        # 不建议在 Dataset 初始化时直接 to(device)
        # 因为 DataLoader 会在 CPU 端工作，一般在训练循环中再搬到 GPU
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
    

def loaddata(path, batch_size=64, shuffle=True):
        """
        加载数据集并返回两个对象：
        1. 原始 dataset（可以用于访问完整数据或调试）
        2. DataLoader（用于 mini-batch 训练）
        
        参数：
            batch_size: 每个 batch 的大小
            shuffle: 是否在每个 epoch 开始前打乱数据
        """
        dataset = MnistDataSet(path)
        
        # ⭐ 关键修改：真正使用 batch_size 和 shuffle
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return dataset, dataloader
