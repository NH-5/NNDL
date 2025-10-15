import torch


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def sigmoid(z):
    return 1.0 / (1.0 + torch.exp(-z))


class FNNet(object):
    def __init__(self, size):
        """
        size: 一个列表，表示每层的神经元数量，例如 [784, 64, 10]
        """

        self.num_layers = len(size)
        self.size = size


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
        a: 输入向量 (size[0],)
        返回：输出向量 (size[-1],1)
        """
        a = a.view(-1, 1)  # 转列向量
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(torch.matmul(w, a) + b)
        return a



    def SGD(self, train_data, epochs, lr, test_data=None):
        """
        train_data: DataLoader(已经包含 mini-batch)
        epochs: 训练轮数
        lr: 学习率
        test_data: 测试集 DataLoader,可选
        is_log: 是否写入日志文件
        """
        Loss = []
        Accuracy = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_count = 0


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

        return Loss, Accuracy

    
    def update_parameter(self, batch_x, batch_y, lr):
        # batch_x: (batch_size, size[0])
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

            # one-hot 编码
            y_onehot = torch.zeros(self.size[-1], 1, device=device)
            y_onehot[y.item(), 0] = 1.0

            output = self.feedforward(x)  # (size[-1],1)

            # MSE loss: sum((y_onehot - output)^2)
            #loss += ((y_onehot - output) ** 2).sum()

            # 交叉熵损失函数
            loss += (-(y_onehot * torch.log((output))+ (1 - y_onehot) * torch.log((1 - output)))).sum()

        loss = loss / (1.0 * n)

        loss.backward()

        with torch.no_grad():
            for w in self.weights:
                w -= lr * w.grad
            for b in self.biases:
                b -= lr * b.grad

        return loss.item()

    
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
