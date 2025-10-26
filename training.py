import torch
import torch.nn as nn
import torch.optim as optim
from CNN import CNNet as cnet
from MLP import FNNet as fnet
from dataLoader import PNGLoader as pngl
from huggingface_dataloader import loader_hf_data as lhfd
from modelscope_dataloader import loder_ms_data as lmsd
from notify import bark_send as bs
from logs import write_logs as wl
import argparse


class Network(nn.Module):
    def __init__(self, size, input_channels=3, input_size = (360, 640), is_flatten = False):
        """
        size:list表示fnn输入层外的部分,
            如[2,3,1]表示两个隐藏层2个神经元和3个神经元,1个输入层神经元
        """
        super().__init__()
        cnn = cnet(
            input_channels=input_channels,
            input_size=input_size,
            is_flatten=is_flatten
        )
        out_dim_of_cnn = cnn.get_flatten_dim()
        size_of_fnn = []
        size_of_fnn.append(out_dim_of_cnn)
        size_of_fnn.extend(size)
        fnn = fnet(size=size_of_fnn)

        self.cnn = cnn
        self.fnn = fnn

    def forward(self, x):
        x = self.cnn.forward(x)
        x = self.fnn.forward(x)
        return x
    

def train(model, trainloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch in trainloader:
        features, labels = batch['image:FILE'].to(device), batch['category'].to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * features.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    loss = running_loss / total
    
    return loss


def validate(model, testloader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in testloader:
            features, labels = batch['image:FILE'].to(device), batch['category'].to(device)
            
            
            outputs = model(features)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    acc = correct / total
    
    return acc


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type = float, default= 1e-3)
    parser.add_argument('--batch_size' , type = int, default= 16)
    parser.add_argument('--epochs', type= int, default= 30)
    parser.add_argument('--cpu', action='store_false', help='出现时为False,True时使用cuda,False时使用cpu')
    parser.add_argument('--mutilgpu', action='store_true', help='是否使用多卡训练,出现时为True,True时使用多卡')
    parser.add_argument('--streaming', action='store_true', help='使用huggingface或modelscope接口加载数据集时是否流式传输,出现时True表示使用流式传输')

    return parser.parse_args()


def main():

    args = get_args()

    device = torch.device('cuda' if torch.cuda.is_available() and args.cpu else 'cpu')
    # 超参数
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    use_multi_gpu = args.mutilgpu
    losses = []
    accuracys = []


    model = Network(
        size=[32,100],
        input_channels=3,
        input_size=(384,512),
        is_flatten=True
    )

    if use_multi_gpu and torch.cuda.device_count() > 1:
        batch_size = batch_size * torch.cuda.device_count()
        model = nn.DataParallel(model)


    trainloader, testloader = lmsd(
        MSDataSet='tany0699/mini_imagenet100',
        batch_size=batch_size,
        is_streaming=args.streaming,
        shuffle=True
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    

    for epoch in range(epochs):
        loss = train(
            model=model,
            trainloader=trainloader,
            optimizer=optimizer,
            criterion=criterion,
            device=device
        )
        accuracy = validate(
            model=model,
            testloader=testloader,
            device=device
        )
        print(f"Epoch {epoch+1} : loss {loss} accuracy {accuracy}.")
        losses.append(loss)
        accuracys.append(accuracy)
    

    wl(losses, accuracys, epochs, batch_size, lr)
    bs('训练结束',f'Epoch {epochs} : loss {losses[-1]} accuracy {accuracys[-1]}.')

if __name__ == '__main__':
    main()