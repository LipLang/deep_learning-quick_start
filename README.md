# deep_learning-quick_start -- `深度学习`起步集

### 1. 框架选择
- 再次斟酌之后, 决定all-in `PyTorch`, 其killing feature是`动态计算图`:
    * 动态计算图是在代码运行时，根据实际执行的操作动态构建的。每次运行代码时，计算图都可能不同，这取决于输入数据和程序控制流。Pytorch运行代码时自动跟踪所有操作，并自动计算梯度。
    * 而静态计算图是在代码编译阶段就预先定义好的，不会在运行时改变。TensorFlow/JAX在编译阶段就计算好梯度函数，并在运行时调用。
    * 动态图使用标准 Python 控制流 (if/else,for,while) ，直观易懂，方便调试。静态图需用框架提供的特殊控制流操作 (jax.lax.cond, tf.cond)，学习成本高，代码可读性低, 调试困难。
- 相比之下, `TensorFlow`和`JAX`都主要是面向静态计算图的.
- 这样看来, 谷歌在深度学习上没踩对拍子, 纰漏的根源在哪里?

### 2. PyTorch的DL框架模板
``` Python
import enum, time, torch
import numpy as np
from typing import NewType
from torch import nn, optim
from einops import rearrange
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

EPSILON = 1e-7
ACTIVATIONS = ["ReLU", "RReLU", "PReLU", "ELU", "Softplus", "Tanh", "SELU", "LeakyReLU", "Sigmoid", "GELU",]
POOLS = ["MaxPool1d", "AvgPool1d",]
NORMS = ["BatchNorm1d", "LayerNorm",]

def get_nn_attr(attr, checklist=[]):
    if not attr: # attr in {'', None, [], 0, ...}
        return nn.Identity
    else:
        if checklist: assert attr in checklist, f"'{attr}' is not in {checklist}"
        return getattr(nn, attr)

# fn = get_nn_attr('ReLU') -> fn would be the activation func.

# TODO: 根据需要导入其他库，如 sklearn 用于数据处理，pandas 用于数据加载等

# 辅助函数，用于打印时间戳和状态信息
def get_time(): return time.strftime("\n[%C%y-%m-%d =%W=%w= %H:%M:%S]", time.localtime())
def ready(txt=''): print(get_time() + f' {txt}: Ready2Go')
def done(txt=''):  print(get_time() + f' {txt}: GetThere')
#

# 1. 定义数据集
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    # TODO: 如果需要数据预处理或增强，可以在这里添加相关方法

# 2. 定义模型
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # 定义模型层
        # TODO: 修改这里的网络结构以适应您的问题
        self.layers = nn.Sequential(
            nn.Linear(10, 64),  # 输入特征数为10，可以根据实际情况调整
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)    # 输出维度为1，可以根据实际情况调整
        )

    def forward(self, x):
        return self.layers(x)

# 3. 训练函数
def train(model, train_loader, criterion, optimizer, device, epochs=10):
    model.train()  # 设置模型为训练模式
    total_loss = 0
    for epoch in range(epochs):
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)  # 将数据移到指定设备（GPU或CPU）
            optimizer.zero_grad()  # 清零梯度
            outputs = model(X)     # 前向传播
            loss = criterion(outputs, y)  # 计算损失
            loss.backward()        # 反向传播
            optimizer.step()       # 更新参数
            total_loss += loss.item()
        # 打印每个 epoch 的损失
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')
    return total_loss / len(train_loader)

# 4. 评估函数
def evaluate(model, test_loader, criterion, device):
    model.eval()  # 设置模型为评估模式
    total_loss = 0
    with torch.no_grad():  # 在评估时不需要计算梯度
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)
            total_loss += loss.item()
    return total_loss / len(test_loader)

# TODO: 可以添加其他辅助函数，如数据可视化、模型保存加载等

# 5. 主函数
def main():
    # 设置随机种子以确保结果可复现
    torch.manual_seed(42)
    np.random.seed(42)

    # 设置计算设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # TODO: 加载实际数据
    # 这里使用随机生成的示例数据
    X = np.random.randn(1000, 10)  # 1000个样本，每个样本10个特征
    y = np.sum(X, axis=1, keepdims=True)  # 简单的目标值：特征之和

    # 划分训练集和测试集
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # 创建数据加载器
    train_dataset = CustomDataset(X_train, y_train)
    test_dataset = CustomDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 初始化模型
    model = MyModel().to(device)
    print(model)  # 打印模型结构

    # 定义损失函数和优化器
    criterion = nn.MSELoss()  # TODO: 根据任务选择合适的损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # TODO: 调整学习率

    # TODO: 可以在这里添加学习率调度器
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # 训练模型
    num_epochs = 100  # TODO: 调整训练轮数
    train_losses, test_losses = [], []

    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        test_loss = evaluate(model, test_loader, criterion, device)
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        # TODO: 如果使用学习率调度器，在这里更新
        # scheduler.step()

        if (epoch + 1) % 10 == 0:  # 每10轮打印一次结果
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

        # TODO: 可以在这里添加早停逻辑

    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Test Loss')
    plt.show()

    # 保存模型
    torch.save(model.state_dict(), 'model.pth')
    print("Model saved to 'model.pth'")

    # TODO: 可以添加模型评估代码，如计算测试集上的准确率、F1分数等

if __name__ == "__main__": main()
```

### 3. 落脚方向
`时间序列分析`: 当大佬们集中于`通用智能模型`的时候, 我觉得`超越大脑`或许也是有趣的方向~
