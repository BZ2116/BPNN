# BPNN
BPNN学习以及练习
github代码仓库
1. 鸢尾花（iris）数据集上的分类（Classification）
代码详解（首次学习，最详细）

---
引用库
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
1. import torch
 导入 PyTorch 库，这是一个常用的深度学习框架，主要用于张量计算和构建神经网络模型。
2. import torch.nn as nn
 导入 PyTorch 的神经网络模块，并将其命名为 nn。这个模块中包含了各种构建神经网络所需的层（如全连接层、卷积层）、激活函数以及损失函数等。
3. import torch.optim as optim
 导入 PyTorch 的优化器模块，并将其命名为 optim。该模块提供了多种优化算法（如随机梯度下降 SGD、Adam 等），用于更新模型参数以最小化损失函数。
4. from torch.utils.data import DataLoader, TensorDataset
- TensorDataset：用于将多个张量打包成一个数据集，通常用来存储特征和标签。
- DataLoader：用于将数据集按批次加载，支持数据的随机打乱、并行加载等功能，从而方便模型训练时的数据迭代。
5. from sklearn.datasets import load_iris
 导入 scikit-learn 库中的 load_iris 函数，用于加载鸢尾花（Iris）数据集。鸢尾花数据集是机器学习中非常经典的多类别分类数据集，常用于算法示例和教学。load_iris
6. from sklearn.model_selection import train_test_split
 导入数据集划分工具 train_test_split，该函数可以将数据集随机分成训练集和测试集，便于模型训练和验证。train_test_split
7. from sklearn.preprocessing import StandardScaler
 导入 StandardScaler，用于对数据进行标准化处理。它通过去除均值并缩放到单位方差，帮助提升模型训练的稳定性和收敛速度。StandardScaler

---
加载数据
这个数据集是机器学习领域中经典的样本数据集，共包含 150 个样本，每个样本有 4 个特征，并且样本被划分为 3 个类别。

iris = load_iris()
X = iris.data# 特征数据 (形状：150个样本 × 4个特征)
y = iris.target# 标签数据 (0,1,2)
X 通常是一个二维的 NumPy 数组，形状为 (150, 4)，即 150 个样本，每个样本包含 4 个特征。
这 4 个特征分别对应鸢尾花的：花萼长度、花萼宽度、花瓣长度和花瓣宽度。
y 是一个一维的 NumPy 数组，包含了 150 个元素。
数组中的每个值都是一个整数，通常为 0、1 或 2，每个数字代表一种鸢尾花的类别。

---
数据预处理
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
1. 将数据分为训练集（80%）和测试集（20%），确保模型训练后能验证效果。
2. 创建了一个 StandardScaler 的实例
作用： StandardScaler 会对数据进行标准化处理，即将数据的均值变为 0，方差缩放为 1。这种预处理有助于提升机器学习模型的训练效果，因为标准化可以防止特征间因量纲不同而影响模型的学习。
3. fit_transform 方法首先对训练数据 X_train 进行拟合（fit），计算出每个特征的均值和标准差，然后再使用这些统计量对训练数据进行转换（transform）。
目的：
 通过标准化操作，将每个特征的数据调整为均值为 0、标准差为 1 的分布。这有助于提高模型的训练稳定性和收敛速度，因为它消除了不同特征量纲不同或数值范围相差较大的问题。
4. 这里使用 transform 方法将测试集 X_test 转换为标准化后的数据，但不重新计算均值和标准差。
原因：
 测试集需要使用在训练集上计算得到的均值和标准差进行转换，这样可以保证训练和测试的数据在同一个标准化尺度上，从而确保模型评估的准确性。

---
转换为Tensor
PyTorch处理的数据必须是张量（多维数组）。
X_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.LongTensor(y_test)
1. 将经过标准化处理后的 NumPy 数组 X_train 转换为 PyTorch 的 FloatTensor。
原因： 深度学习模型通常要求输入为浮点数类型，且 PyTorch 默认使用 32 位浮点数进行计算。
2. 将训练集标签 y_train 转换为 PyTorch 的 LongTensor。
原因： 对于分类任务，损失函数（例如 nn.CrossEntropyLoss）要求标签为整数型（长整型），表示类别索引。
3,4同上

---
创建DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)
1. 使用 TensorDataset 将训练数据的特征 X_train 和标签 y_train 组合成一个数据集。
每个数据样本将以 (特征, 标签) 的元组形式存储，便于后续的批量加载和训练。
2. 使用 DataLoader 将训练数据集划分为小批量（mini-batch）数据，并提供迭代器接口，便于模型训练。
参数解析：
- batch_size=10：每个批次包含 10 个样本，帮助平衡内存使用和训练效率。
- shuffle=True：在每个 epoch 开始时打乱数据顺序，有助于模型更好地泛化，避免因数据顺序而产生偏差。
3. 同样地，将测试集的特征 X_test 和标签 y_test 组合成一个数据集，供模型评估使用。
每个样本也以 (特征, 标签) 的形式存储，确保测试数据的结构与训练数据一致。
4. 使用 DataLoader 为测试数据集创建一个加载器，用于按批次提取数据进行评估。
参数解析：
- batch_size=10：每个批次包含 10 个样本。
- shuffle=False：测试阶段不需要打乱数据，保持数据顺序的一致性便于评估和结果的复现。

---
 定义BPNN模型
class BPNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BPNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
1. 定义了一个名为 BPNN 的类，并继承自 nn.Module。
继承 nn.Module 是构建 PyTorch 模型的基础，使得该类可以利用 PyTorch 提供的模型构建、参数管理和自动求导功能。
2. 初始化方法 init
参数说明：
- input_size：输入数据的特征维度（即输入层神经元个数）。
- hidden_size：隐藏层中神经元的个数。
- output_size：输出层的神经元个数，通常对应分类任务中的类别数。
3. super(BPNN, self).__init__()： 调用了父类（nn.Module）的初始化方法，确保模型内部的所有模块和参数正确初始化。
4. self.fc1 = nn.Linear(input_size, hidden_size)：定义了第一层全连接层，将输入数据从 input_size 维映射到 hidden_size 维。
  执行的是线性变换：y = xW^T + b。
5. self.relu = nn.ReLU()：定义了激活函数 ReLU（Rectified Linear Unit），用于在第一层输出后引入非线性，提升模型拟合复杂非线性关系的能力。
6. self.fc2 = nn.Linear(hidden_size, output_size)：定义了第二层全连接层，将隐藏层的输出映射到 output_size 维，通常用于生成最终的预测结果（例如分类任务中的 logits）。
9. out = self.fc1(x)：将输入 x 通过第一层全连接层，得到线性变换的结果。
10. out = self.relu(out)：对线性变换后的结果应用 ReLU 激活函数，使得模型能够学习非线性关系。
11. out = self.fc2(out)：将激活后的数据通过第二层全连接层，映射到输出空间，得到最终的预测输出。

---
初始化模型、损失函数和优化器
model = BPNN(input_size=4, hidden_size=5, output_size=3)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
1. 创建了一个 BPNN 模型实例。
  参数说明：
  - input_size=4：输入层的神经元数量为 4，对应鸢尾花数据集中每个样本的 4 个特征。
  - hidden_size=5：隐藏层中神经元数量为 5，这是一个可调参数，影响模型学习能力。
  - output_size=3：输出层神经元数量为 3，对应鸢尾花数据集中 3 个类别的预测。
  意义： 该模型结构简单，包含一层隐藏层，适用于解决多类别分类问题。
2. 定义了交叉熵损失函数，用于衡量模型预测与真实标签之间的误差。
  特点：
  - 内部会先对模型输出应用 LogSoftmax，然后计算负对数似然损失（NLLLoss）。
  - 适用于多类别分类任务，其目标是最小化预测类别概率分布与真实分布之间的差异。
3. 使用 Adam 优化算法来更新模型的参数，以最小化损失函数。
  参数说明：
  - model.parameters()：指定了模型中所有需要优化的参数。
  - lr=0.01：设置学习率为 0.01，控制参数更新的步长。
  特点： Adam 优化器结合了动量和自适应学习率调整，通常能带来较快的收敛速度和较好的性能。

---
训练模型
num_epochs = 100
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
1. 定义整个训练过程要遍历整个训练数据集的次数为 100 轮（epoch）
2. 外层循环遍历每个 epoch：每个 epoch 表示对整个训练数据集进行一次完整的遍历。
3. 内层循环遍历每个 mini-batch：使用 train_loader 按照设置的 batch_size（这里为 10）依次加载数据，每个 inputs 是一个小批次的特征，labels 是对应的小批次标签。
4. 前向传播：将当前批次的输入数据 inputs 传入模型，得到预测输出 outputs。这一步中，数据依次经过第一层全连接、ReLU 激活函数和第二层全连接，生成 logits。
5. 计算损失：利用交叉熵损失函数 criterion 计算模型预测输出 outputs 与真实标签 labels 之间的误差。交叉熵损失适用于多类别分类任务，内部会对 logits 进行 softmax 操作。
6. 梯度清零：在每个 mini-batch 开始前，将之前累积的梯度信息清零，避免梯度累加对参数更新产生影响。
7. 反向传播：通过反向传播算法计算损失函数关于模型参数的梯度，即自动求导。
8. 更新模型参数：根据计算得到的梯度，使用 Adam 优化器更新模型参数，使得损失函数逐步下降。
9. 每经过 10 个 epoch 后，打印一次当前的训练进度和最后一个 mini-batch 的损失值。

---
测试模型
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print(f'测试准确率: {100 * correct / total:.2f}%')
1. 设置模型为评估模式：将模型切换到评估模式。在评估模式下，诸如 dropout、batch normalization 等层的行为会与训练模式不同（例如 dropout 将关闭），确保评估结果的稳定性。
2. 禁用梯度计算：在这个代码块中，PyTorch 将不会计算梯度。这不仅降低了内存消耗，同时也提高了推理速度，因为在测试阶段不需要梯度信息。
3. 初始化统计变量：correct 用于累计预测正确的样本数。total 用于累计总样本数。这两个变量将在后续计算模型的整体测试准确率。
6. 模型预测：这里的 outputs 通常是未经过 softmax 的 logits，每一行代表一个样本在各个类别上的原始得分。
7. 获取预测结果：使用 torch.max 函数在维度 1（即各个类别得分的维度）上找到最大值，返回两个值：最大值和对应的索引。
  解释： 下划线 _ 表示忽略最大值，只保留 predicted（预测的类别索引），这就是模型认为最可能的类别。
8. 更新统计数据：
  labels.size(0)： 返回当前批次的样本数量，累计到 total 中。
  (predicted == labels)： 对比模型预测与真实标签，返回一个布尔张量，其中 True 表示预测正确。
  .sum().item()： 将布尔张量中 True 的数量求和，并转换为 Python 数值，累计到 correct 中。
