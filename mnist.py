import torch
from torch.utils.data import Dataset, DataLoader    
import pandas as pd

device = 'cuda' if torch.cuda.is_available() else 'cpu'


train_data_path = 'data/mnist_train.csv'
test_data_path = 'data/mnist_test.csv'


def sigmoid(z):
    return 1.0 / (1.0 + torch.exp(-z))

def sigmoid_prime(z):
    """Sigmoid 函数的导数"""
    return sigmoid(z) * (1 - sigmoid(z))


class Network(object):
    def __init__(self, size):
        self.num_layers = len(size)
        self.size = size
        self.biases = [torch.randn(y, 1, requires_grad=True, device=device) for y in size[1:]]
        self.weights = [torch.randn(y, x, requires_grad=True, device=device) for x, y in zip(size[:-1],size[1:])]
        

    def feedforward(self, a:torch.tensor):
        a = a.view(-1, 1)
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(torch.matmul(w, a) + b)
        return a
    

    def SGD(
            self,
            train_data:torch.tensor,
            epochs:int,
            batch_size:int,
            lr:float,
            test_data:torch.tensor=None,
    ):
        """
        train_data:torch.tensor的结构如下
        train_data = [
            [ x[0],y[0] ],
            [ x[1],y[1] ],
            [ x[2],y[2] ],
            ...
        ]
        对于第i条数据train_data[i]有
        train_data[i][0] = x[i]:torch.tensor
        train_data[i][1] = y[i]:torch.tensor
        """
        n = len(train_data)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_count = 0
            for batch_x, batch_y in train_loader:
                mini_batch = [(batch_x[i].to(device), batch_y[i].to(device)) 
                              for i in range(len(batch_x))]
                batch_loss = self.update_parameter(mini_batch, lr)
                epoch_loss += batch_loss
                batch_count += 1

            avg_loss = epoch_loss / batch_count
            print(f"Epoch {epoch + 1}: Loss = {avg_loss:.6f}", end="")
            
            if test_data:
                self.evaluate(test_data, epoch)
            else:
                print()

    
    def update_parameter(self, datas, lr):
        n = len(datas)

        for w in self.weights:
            w.grad = None

        for b in self.biases:
            b.grad = None

        loss = 0
        for data in datas:
            x = data[0]
            y = data[1]
            y_onehot = torch.zeros(self.size[-1], 1, device=device)
            y_onehot[y.item(), 0] = 1.0
            loss += ((y_onehot - self.feedforward(x)) ** 2).sum()
        loss = loss / (2.0 * n)
        

        loss.backward()

        with torch.no_grad():
            for w in self.weights:
                w -= lr * w.grad
            for b in self.biases:
                b -= lr * b.grad

        return loss.item()
        

    def evaluate(self, test_set, epoch):
        n_test = len(test_set)
        current = 0
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
        for data in test_loader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            output = self.feedforward(x)
            predicted_label = torch.argmax(output).item()
            true_label = y.item()
            if abs(predicted_label-true_label) <= 10 ** (-6):
                current += 1
        accuracy = current / n_test
        print(f", Accuracy = {accuracy:.4f}")

class LoadData(Dataset):
    def __init__(self, file_path):
        df = pd.read_csv(file_path)
        self.labels = torch.tensor(df.iloc[:, 0].values,dtype=torch.long,device=device)
        self.features = torch.tensor(df.iloc[:, 1:].values,dtype=torch.float32,device=device) / 255.0

    def __len__(self):
        return len(self.labels)
    

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


if __name__ == '__main__':
    train_set = LoadData(train_data_path)
    test_set = LoadData(test_data_path)

    net = Network([784, 64, 10])
    net.SGD(train_data=train_set, epochs=30, batch_size=64, lr=5, test_data=test_set)
