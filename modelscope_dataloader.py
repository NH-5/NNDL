from modelscope.msdatasets import MsDataset
from torchvision import transforms as tf
from torch.utils.data import DataLoader
from PIL import Image
import torch

def preprocess(example):
    img = Image.open(example['image:FILE']).convert('RGB')
    example['image:FILE'] = transform(img)
    example['category'] = torch.tensor(example['category'])
    return example

transform = tf.Compose([
    tf.Resize((384,512)),
    tf.ToTensor()
])

def loder_ms_data(MSDataSet, batch_size, is_streaming = True, shuffle = True):

    

    if is_streaming:
        """
        先用小数据集测试非流式传输模式的代码,流式传输等待未来开发工作
        """
        pass

    else:
        traindata = MsDataset.load(MSDataSet,split='train')
        #testdata = MsDataset.load(MSDataSet,split='test')
        validationdata = MsDataset.load(MSDataSet, split='validation')

        traindata = traindata._to_torch_dataset_with_processors(preprocess)
        #testdata = testdata._to_torch_dataset_with_processors(preprocess)
        validationdata = validationdata._to_torch_dataset_with_processors(preprocess)

        trainloader = DataLoader(traindata,batch_size=batch_size,shuffle=shuffle)
        #testloader = DataLoader(testdata,batch_size=batch_size,shuffle=shuffle)
        validationloader = DataLoader(validationdata,batch_size=batch_size,shuffle=shuffle)
        

    return trainloader, validationloader