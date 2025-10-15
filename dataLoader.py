import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
from PIL import Image
import os

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


class PNGDataSet(Dataset):
    """
    当前所使用的数据集均为640*360的png图片(RGBA模式)
    """
    def __init__(self, path):
        """
        同一张图片对应于featurns[i],labels[i]
        """
        dict = {
            "Among Us":torch.tensor([1,0,0,0,0,0,0,0,0,0],dtype=torch.int),
            "Apex Legends":torch.tensor([0,1,0,0,0,0,0,0,0,0],dtype=torch.int),
            "Fortnite":torch.tensor([0,0,1,0,0,0,0,0,0,0],dtype=torch.int),
            "Forza Horizon":torch.tensor([0,0,0,1,0,0,0,0,0,0],dtype=torch.int),
            "Free Fire":torch.tensor([0,0,0,0,1,0,0,0,0,0],dtype=torch.int),
            "Genshin Impact":torch.tensor([0,0,0,0,0,1,0,0,0,0],dtype=torch.int),
            "God of War":torch.tensor([0,0,0,0,0,0,1,0,0,0],dtype=torch.int),
            "Minecraft":torch.tensor([0,0,0,0,0,0,0,1,0,0],dtype=torch.int),
            "Roblox":torch.tensor([0,0,0,0,0,0,0,0,1,0],dtype=torch.int),
            "Terraria":torch.tensor([0,0,0,0,0,0,0,0,0,1],dtype=torch.int)
        }
        labels = []
        features = []
        to_tensor = transforms.ToTensor()
        for game in os.listdir(path):
            game_path = os.path.join(path, game)
            for imgname in os.listdir(game_path):
                with Image.open(os.path.join(game_path, imgname)).convert('RGB') as img:
                    img = to_tensor(img)
                    labels.append(dict[game])
                    features.append(img)
                    breakpoint()
        self.features = features
        self.labels = labels


    def __len__(self):
        return len()
    

    def __getitem__(self, index):
         return self.features[index], self.labels[index]
    

def PNGLoader(path, batch_size=64, shuffle=False):
    dataset = PNGDataSet(path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataset, dataloader

if __name__ == '__main__':
    trainset = PNGDataSet('data/gameplay-images/train_data')
