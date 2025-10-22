from datasets import load_dataset
from torchvision import transforms as tf
from torch.utils.data import DataLoader, IterableDataset
from PIL import Image

class TransformIterableDataset(IterableDataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform
    
    def __iter__(self):
        for batch in self.dataset:
            image = batch['image']
            if isinstance(image, Image.Image):
                image = image.convert('RGB')
            
            image = self.transform(image)
            label = batch['label']
            yield image, label

def loader_hf_data(HFDataSet, batch_size):
    """
    使用huggingface上的ImageNet数据集
    """
    traindata = load_dataset(HFDataSet, split='train', streaming=True)
    testdata = load_dataset(HFDataSet, split='test', streaming=True)
    validationdata = load_dataset(HFDataSet, split='validation', streaming=True)

    transform = tf.Compose([
        tf.Resize((384,512)),
        tf.ToTensor()
    ])

    # 使用自定义包装器应用转换
    traindata = TransformIterableDataset(traindata, transform)
    testdata = TransformIterableDataset(testdata, transform)
    validationdata = TransformIterableDataset(validationdata, transform)

    # 注意：IterableDataset 不支持 shuffle=True
    trainloader = DataLoader(traindata, batch_size=batch_size)
    testloader = DataLoader(testdata, batch_size=batch_size)
    validationloader = DataLoader(validationdata, batch_size=batch_size)

    return trainloader, testloader, validationloader