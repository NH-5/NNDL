from FNN import FNNet
import dataLoader
from notify import bark_send
from logs import write_logs as wl

train_data_path = 'data/gameplay-images/train_data'
test_data_path = 'data/gameplay-images/test_data'


if __name__ == '__main__':

    _, train_loader = dataLoader.PNGLoader(train_data_path, batch_size=64, shuffle=True)
    _, test_loader = dataLoader.PNGLoader(test_data_path, batch_size=64, shuffle=False)

    net = FNNet([784, 64, 10])

    loss, accuracy = net.SGD(train_data=train_loader, epochs=30, lr=0.01, test_data=test_loader)
    wl(Loss=loss, Accuracy=accuracy , epochs=30)
    bark_send("训练完成", f"训练结束！最终 Loss: {loss[-1]:.4f}, Accuracy: {accuracy[-1]:.4f}")