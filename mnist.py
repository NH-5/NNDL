import torch
import pandas as pd
from datetime import datetime, timezone, timedelta
import os
import dataLoader
from notify import bark_send

# 自动检测设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_data_path = 'data/mnist_train.csv'
test_data_path = 'data/mnist_test.csv'


# =============================
# 激活函数及其导数
# =============================
def sigmoid(z):
    return 1.0 / (1.0 + torch.exp(-z))

def sigmoid_prime(z):
    """Sigmoid 函数的导数（此项目中未使用，但保留）"""
    return sigmoid(z) * (1 - sigmoid(z))


# =============================
# 网络结构定义
# =============================
class Network(object):
    def __init__(self, size):
        """
        size: 一个列表，表示每层的神经元数量，例如 [784, 64, 10]
        """

        self.num_layers = len(size)
        self.size = size

        # ===============================
        # ✅ 正确初始化权重和偏置（关键修复）
        #
        # 之前的问题：
        #   torch.randn(..., requires_grad=True) * 0.01
        # 会导致结果变成非叶子张量，grad 不会保存
        #
        # ✅ 现在的解决方案：
        #   先创建叶子张量（requires_grad=True），再用 .data 乘缩放因子
        #   这样既保持叶子张量身份，又能控制权重大小
        # ===============================

        # 初始化权重 weights
        self.weights = []
        for x, y in zip(size[:-1], size[1:]):
            # 创建叶子张量
            w = torch.randn(y, x, device=device, requires_grad=True)
            # 用 .data 缩小权重，避免 sigmoid 饱和
            w.data *= 0.01
            self.weights.append(w)

        # 初始化偏置 biases
        self.biases = []
        for y in size[1:]:
            b = torch.randn(y, 1, device=device, requires_grad=True)
            b.data *= 0.01
            self.biases.append(b)


    def feedforward(self, a: torch.tensor):
        """
        前向传播
        a: 输入向量 (784,)
        返回：输出向量 (10,1)
        """
        a = a.view(-1, 1)  # 转列向量
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(torch.matmul(w, a) + b)
        return a

    # =============================
    # SGD 训练
    # =============================
    def SGD(self, train_data, epochs, lr, test_data=None, is_log=False):
        """
        train_data: DataLoader（已经包含 mini-batch）
        epochs: 训练轮数
        lr: 学习率（建议<=0.1）
        test_data: 测试集 DataLoader，可选
        is_log: 是否写入日志文件
        """
        Loss = []
        Accuracy = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_count = 0

            # ⭐⭐ 关键点：train_data 是 DataLoader，提供 (batch_x, batch_y)
            for batch_x, batch_y in train_data:
                # 把数据移动到 GPU（如果有）
                batch_x = batch_x.to(device)  # shape: (batch_size, 784)
                batch_y = batch_y.to(device)  # shape: (batch_size,)

                # 调用 update_parameter 计算损失并更新参数
                batch_loss = self.update_parameter(batch_x, batch_y, lr)
                epoch_loss += batch_loss
                batch_count += 1

            avg_loss = epoch_loss / batch_count
            Loss.append(avg_loss)
            print(f"Epoch {epoch + 1}: Loss = {avg_loss:.6f}", end="")

            # 若提供 test_data，则评估准确率
            if test_data:
                accuracy = self.evaluate(test_data)
                Accuracy.append(accuracy)
            else:
                print()

        self.Loss = Loss
        self.Accuracy = Accuracy
        self.epochs = epochs
        if is_log:
            self.log()

        return Loss, Accuracy

    # =============================
    # 参数更新（反向传播 + 手动SGD）
    # =============================
    def update_parameter(self, batch_x, batch_y, lr):
        # batch_x: (batch_size, 784)
        # batch_y: (batch_size,)
        n = batch_x.shape[0]  # batch_size

        # 清空梯度
        for w in self.weights:
            if w.grad is not None:
                w.grad = None
        for b in self.biases:
            if b.grad is not None:
                b.grad = None

        # 使用 MSE 手动计算损失
        loss = 0
        # 逐样本累加损失
        for i in range(n):
            x = batch_x[i]
            y = batch_y[i]

            # one-hot 编码真实类别
            y_onehot = torch.zeros(self.size[-1], 1, device=device)
            y_onehot[y.item(), 0] = 1.0

            # 前向传播
            output = self.feedforward(x)  # (10,1)

            # MSE loss: sum((y_onehot - output)^2)
            loss += ((y_onehot - output) ** 2).sum()

        # 平均损失除以 2n（经典公式）
        loss = loss / (2.0 * n)

        # 反向传播
        loss.backward()

        # 使用学习率更新参数
        with torch.no_grad():
            for w in self.weights:
                w -= lr * w.grad
            for b in self.biases:
                b -= lr * b.grad

        return loss.item()

    # =============================
    # 测试准确率
    # =============================
    def evaluate(self, test_set):
        n_test = 0
        current = 0

        # test_set 也是 DataLoader，每个元素是一个 batch
        for batch_x, batch_y in test_set:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # 遍历 batch 内每个样本
            for i in range(batch_x.shape[0]):
                x = batch_x[i]
                y = batch_y[i]

                output = self.feedforward(x)
                predicted_label = torch.argmax(output).item()
                true_label = y.item()

                if predicted_label == true_label:
                    current += 1
                n_test += 1

        accuracy = current / n_test
        print(f", Accuracy = {accuracy:.4f}")
        return accuracy

    # =============================
    # 记录日志文件
    # =============================
    def log(self):
        utc_plus_8 = timezone(timedelta(hours=8))
        now_utc_8 = datetime.now(utc_plus_8)
        formatted_time = now_utc_8.strftime('%Y-%m-%d %H:%M:%S UTC+8')
        filename = f'{formatted_time}.txt'
        logs_path = os.path.join("logs/", filename)
        with open(logs_path, 'a') as file:
            file.write(f"Training Time is {formatted_time}.\n")
            for epoch in range(self.epochs):
                file.write(
                    f"Epoch {epoch} : Loss is {self.Loss[epoch]}, Accuracy is {self.Accuracy[epoch]}.\n"
                )


# =============================
# 主程序入口
# =============================
if __name__ == '__main__':
    # 使用我们改好的 loaddata（已支持 batch_size 和 shuffle）
    _, train_loader = dataLoader.loaddata(train_data_path, batch_size=64, shuffle=True)
    _, test_loader = dataLoader.loaddata(test_data_path, batch_size=64, shuffle=False)

    # 初始化网络
    net = Network([784, 64, 10])

    # ****** 关键改动：学习率降低为 0.1 （或 0.01 更安全） ******
    loss, accuracy = net.SGD(train_data=train_loader, epochs=30, lr=3, test_data=test_loader, is_log=True)
    bark_send("训练完成", f"训练结束！最终 Loss: {loss[-1]:.4f}, Accuracy: {accuracy[-1]:.4f}")
