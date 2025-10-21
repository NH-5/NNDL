from datasets import load_dataset
from torchvision import transforms as tf
from torch.utils.data import DataLoader

def loader_hf_data(HFDataSet, batch_size, shuffle=False):
    """
    使用huggingface上的ImageNet数据集
    """
    traindata = load_dataset(HFDataSet, split='train', streaming=True)
    testdata = load_dataset(HFDataSet, split='test', streaming=True)
    validationdata = load_dataset(HFDataSet, split='validation', streaming=True)

    transform = tf.Compose([
        tf.ToTensor()
    ])


    traindata.set_transform(lambda batch:{
        'image':[transform(img) for img in batch['image']],
        'label':batch['label']
    })

    testdata.set_transform(lambda batch:{
        'image':[transform(img) for img in batch['image']],
        'label':batch['label']
    })

    validationdata.set_transform(lambda batch:{
        'image':[transform(img) for img in batch['image']],
        'label':batch['label']
    })

    trainloader = DataLoader(traindata, batch_size=batch_size, shuffle=shuffle)
    testloader = DataLoader(testdata, batch_size=batch_size, shuffle=False)
    validationloader = DataLoader(validationdata, batch_size=batch_size, shuffle=False)

    return trainloader, testloader, validationloader



