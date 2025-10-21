import torch
import torch.nn as nn
import torch.optim as optim
from CNN import CNNet as cnet
from MLP import FNNet as fnet
from dataLoader import PNGLoader as pngl
from notify import bark_send as bs
from logs import write_logs as wl


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
    
    for features, labels in trainloader:
        features, labels = features.to(device), labels.to(device)

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
        for features, labels in testloader:
            features, labels = features.to(device), labels.to(device)
            
            
            outputs = model(features)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    acc = correct / total
    
    return acc


if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    trainpath = 'data/gameplay-images/train_data'
    testpath = 'data/gameplay-images/test_data'

    batch_size = 32

    trainloader, testloader = pngl(
        trainpath=trainpath,
        testpath=testpath,
        batch_size=batch_size,
        shuffle=True
    )

    model = Network(
        size=[32,10],
        input_channels=3,
        input_size=(360,640),
        is_flatten=True
    )

    epochs = 30
    lr = 0.01
    use_multi_gpu = False
    losses = []
    accuracys = []

    if use_multi_gpu and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

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
        print(f"Epoch {epoch+1} : loss {loss} accuracy {accuracy}.\n")
        losses.append(loss)
        accuracys.append(accuracy)
    

    wl(losses, accuracys, epochs, batch_size, lr)
    bs('训练结束',f'Epoch {epochs} : loss {losses[-1]} accuracy {accuracys[-1]}.')

