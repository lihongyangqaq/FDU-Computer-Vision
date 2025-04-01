from copy import deepcopy

import numpy as np
import pickle
import matplotlib.pyplot as plt
import re


def load_cifar10_data(data):
    train_data, train_labels = [], []  # 初始化训练集 前者为图片转化的数据，后者为图片对应的标签
    for i in range(1, 6):
        with open(f"{data}/data_batch_{i}", 'rb') as f:  # 使用python的标准输入库读取文件
            batch = pickle.load(f, encoding='latin1')
            train_data.append(batch['data'])
            train_labels.append(batch['labels'])
    train_data = np.vstack(train_data)  # 将数组垂直堆叠成高维数组
    train_labels = np.hstack(train_labels)

    test_data, test_labels = [], []  # 初始化训练集 前者为图片转化的数据，后者为图片对应的标签
    for i in range(1):
        with open(f"{data}/test_batch", 'rb') as f:  # 使用python的标准输入库读取文件
            batch = pickle.load(f, encoding='latin1')
            test_data.append(batch['data'])
            test_labels.append(batch['labels'])
    test_data = np.vstack(test_data)  # 将数组垂直堆叠成高维数组
    test_labels = np.hstack(test_labels)

    return train_data, train_labels, test_data, test_labels


def preprocess_data(X_train, y_train, X_test, y_test):
    # 归一化到 [0, 1]
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # 将标签转换为 one-hot 编码（假设类别数为 10）
    def to_one_hot(y, num_classes=10):  # 将十进制数0-9转换为one-hot编码 如3->[0,0,0,1,0,0,0,0,0,0]
        one_hot = []
        for Y in y:
            one = [0] * num_classes
            one[Y] = 1
            one_hot.append(one)
        return np.array(one_hot)

    y_train = to_one_hot(y_train)
    y_test = to_one_hot(y_test)
    return X_train, y_train, X_test, y_test


class ThreeLayerNN:
    def __init__(self, input_size, hidden_size, output_size, activation='relu'):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = activation

        # 初始化权重和偏置（用小的随机数）
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        sig = self.sigmoid(x)
        return sig * (1 - sig)

    def softmax(self, z):
        ez = np.exp(z - np.max(z, axis=1, keepdims=True))  # 减去最大值防止指数爆炸
        return ez / np.sum(ez, axis=1, keepdims=True)

    def forward(self, X):
        """前向传播"""
        self.z1 = np.dot(X, self.W1) + self.b1  # 隐藏层线性变换
        if self.activation == 'relu':
            self.a1 = self.relu(self.z1)  # ReLU 激活
        elif self.activation == 'sigmoid':
            self.a1 = self.sigmoid(self.z1)  # Sigmoid 激活
        self.z2 = np.dot(self.a1, self.W2) + self.b2  # 输出层线性变换
        self.a2 = self.softmax(self.z2)
        return self.a2

    def backward(self, X, y_true, learning_rate, reg_lambda=0.01):
        m = X.shape[0]  # 获得样本量
        # delta2的形式简洁是由softmax函数与交叉熵函数的性质配合得到的
        delta2 = self.a2 - y_true  # 真实值与预测值的差 即最后层梯度
        if self.activation == 'relu':
            delta1 = np.dot(delta2, self.W2.T) * self.relu_derivative(self.z1)
        elif self.activation == 'sigmoid':
            delta1 = np.dot(delta2, self.W2.T) * self.sigmoid_derivative(self.z1)

        dW2 = np.dot(self.a1.T, delta2) / m + reg_lambda * self.W2
        dW1 = np.dot(X.T, delta1) / m + reg_lambda * self.W1
        db2 = np.sum(delta2, axis=0, keepdims=True) / m
        db1 = np.sum(delta1, axis=0, keepdims=True) / m

        self.W2 -= learning_rate * dW2
        self.W1 -= learning_rate * dW1
        self.b2 -= learning_rate * db2
        self.b1 -= learning_rate * db1

    def compute_loss(self, X, y_true, reg_lambda=0.01):
        m = X.shape[0]
        y_pred = self.forward(X)

        # 计算交叉熵损失
        cross_entropy = -np.sum(y_true * np.log(y_pred + 1e-8)) / m

        # 计算L2正则化项
        l2_reg = (reg_lambda / (2 * m)) * (np.sum(np.square(self.W1)) + np.sum(np.square(self.W2)))

        total_loss = cross_entropy + l2_reg
        return total_loss

    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=64, learning_rate=0.01, reg_lambda=0.01):
        best_val = 0
        best_weight = None
        train_history = {
            'val_acc': [],
            'train_acc': [],
            'train_loss': [],
            'val_loss': [],
            'epoch': []
        }

        for epoch in range(epochs):
            indices = np.arange(X_train.shape[0])
            np.random.shuffle(indices)

            # Mini-batch训练
            for i in range(0, X_train.shape[0], batch_size):
                train_indices = indices[i:i + batch_size]
                X_batch = X_train[train_indices]
                y_batch = y_train[train_indices]

                self.forward(X_batch)
                self.backward(X_batch, y_batch, learning_rate, reg_lambda)

            # 计算训练集和验证集的指标
            train_acc = self.evaluate(X_train[:1000], y_train[:1000])  # 使用部分训练集评估加速
            val_acc = self.evaluate(X_val, y_val)
            train_loss = self.compute_loss(X_train[:1000], y_train[:1000], reg_lambda)
            val_loss = self.compute_loss(X_val, y_val, reg_lambda)

            # 记录历史数据
            train_history['train_acc'].append(train_acc)
            train_history['val_acc'].append(val_acc)
            train_history['train_loss'].append(train_loss)
            train_history['val_loss'].append(val_loss)
            train_history['epoch'].append(epoch + 1)

            # 保存最佳模型
            if val_acc > best_val:
                best_val = val_acc
                best_weight = {
                    'W1': self.W1.copy(),
                    'b1': self.b1.copy(),
                    'W2': self.W2.copy(),
                    'b2': self.b2.copy()
                }
                print(
                    f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if best_weight:
            self.W1 = best_weight['W1']
            self.b1 = best_weight['b1']
            self.W2 = best_weight['W2']
            self.b2 = best_weight['b2']

        return train_history, best_weight  # 返回训练历史和最佳权重

    def evaluate(self, X, y):
        y_pred = self.forward(X)
        pred_labels = np.argmax(y_pred, axis=1)
        true_labels = np.argmax(y, axis=1)
        accuracy = np.mean(pred_labels == true_labels)
        return accuracy


def plot_learning_curves(all_histories):
    # 创建两个子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    # 绘制准确率曲线
    for history in all_histories:
        params = history['params']
        label = (f"LR={params['learning_rate']}, BS={params['batch_size']}, "
                 f"λ={params['reg_lambda']}, HS={params['hidden_size']}")

        ax1.plot(history['epoch'], history['val_acc'], label=label)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Validation Accuracy')
        ax1.set_title('Validation Accuracy Curves')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True)

    # 绘制损失曲线
    for history in all_histories:
        params = history['params']
        label = (f"LR={params['learning_rate']}, BS={params['batch_size']}, "
                 f"λ={params['reg_lambda']}, HS={params['hidden_size']}")

        ax2.plot(history['epoch'], history['val_loss'], '--', label=f"Val {label}")
        ax2.plot(history['epoch'], history['train_loss'], '-', label=f"Train {label}")
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Training and Validation Loss Curves')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True)

    plt.tight_layout()
    plt.show()


def automatic_parameter_search(X_train, y_train, X_val, y_val):
    hidden_sizes = [128]
    learning_rates = [0.01, 0.05]
    batch_sizes = [64, 128]
    reg_lambdas = [0.01, 0.1]

    all_histories = []
    best_acc = 0
    best_model = None
    best_params = None

    print("\nStarting automatic parameter search...")
    print(f"Parameter space:")
    print(f"- Hidden sizes: {hidden_sizes}")
    print(f"- Learning rates: {learning_rates}")
    print(f"- Batch sizes: {batch_sizes}")
    print(f"- Regularization strengths: {reg_lambdas}")
    print(f"Total combinations: {len(hidden_sizes) * len(learning_rates) * len(batch_sizes) * len(reg_lambdas)}")

    for hidden_size in hidden_sizes:
        for learning_rate in learning_rates:
            for batch_size in batch_sizes:
                for reg_lambda in reg_lambdas:
                    params = {
                        'hidden_size': hidden_size,
                        'learning_rate': learning_rate,
                        'batch_size': batch_size,
                        'reg_lambda': reg_lambda
                    }
                    print(f"\nTraining with: {params}")

                    model = ThreeLayerNN(3072, hidden_size, 10, activation='relu')
                    history, weights = model.train(X_train, y_train, X_val, y_val,
                                                   epochs=100, batch_size=batch_size,
                                                   learning_rate=learning_rate,
                                                   reg_lambda=reg_lambda)

                    # 保存参数和训练历史
                    history['params'] = params
                    all_histories.append(history)

                    val_acc = model.evaluate(X_val, y_val)
                    print(f"Final Validation Accuracy: {val_acc:.4f}")

                    if val_acc > best_acc:
                        best_acc = val_acc
                        best_model = deepcopy(model)
                        best_params = params
                        best_weights = weights  # 保存最佳权重
                        print("New best model found!")

    print("\nParameter search completed!")
    print(f"Best validation accuracy: {best_acc:.4f}")
    print(f"Best parameters: {best_params}")

    # 绘制学习曲线
    plot_learning_curves(all_histories)
    # 在参数搜索结束后打印最佳权重
    # 将最佳权重保存到文件中
    with open('weight.txt', 'w') as f:
        f.write("Best model weights:\n")
        for key, value in best_weights.items():
            f.write(f"{key} shape: {value.shape}\n")
            f.write(f"{key} values:\n")
            np.savetxt(f, value.flatten().reshape(1, -1), delimiter=',', fmt='%.6f')

    # 在控制台打印权重形状信息
    print("\nBest model weights:")
    for key, value in best_weights.items():
        print(f"{key} shape: {value.shape}")
    return best_model, best_params


def manual_parameter_input(X_train, y_train, X_val, y_val):
    print("\nManual parameter input mode")
    print("Please enter the following parameters:")

    try:
        hidden_size = int(input("Hidden layer size (e.g., 128): "))
        learning_rate = float(input("Learning rate (e.g., 0.01): "))
        batch_size = int(input("Batch size (e.g., 64): "))
        reg_lambda = float(input("Regularization strength (e.g., 0.01): "))
        epochs = int(input("Number of epochs (e.g., 20): "))
    except ValueError:
        print("Invalid input! Please enter numeric values.")
        return None, None

    params = {
        'hidden_size': hidden_size,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'reg_lambda': reg_lambda
    }

    print(f"\nTraining with parameters: {params}")
    model = ThreeLayerNN(3072, hidden_size, 10, activation='relu')
    history, weights = model.train(X_train, y_train, X_val, y_val,
                                   epochs=epochs, batch_size=batch_size,
                                   learning_rate=learning_rate,
                                   reg_lambda=reg_lambda)

    # 保存参数和训练历史
    history['params'] = params
    all_histories = [history]

    val_acc = model.evaluate(X_val, y_val)
    print(f"\nTraining completed!")
    print(f"Final Validation Accuracy: {val_acc:.4f}")

    # 绘制学习曲线
    plot_learning_curves(all_histories)

    return model, params


import re

def load_model_from_weights():
    try:
        with open('weight.txt', 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        # 提取权重形状信息
        shapes = {}
        for line in lines:
            if 'shape' in line:
                try:
                    parts = line.split()
                    name = parts[0]
                    left = parts[2].replace('(', '')
                    left = left.replace(',', '')
                    right = parts[3].replace(')', '')
                    shapes[name] = (int(left), int(right))
                except (IndexError, ValueError) as e:
                    print(f"Error parsing shape info: {e}")
                    continue

        # 检查是否所有形状信息都已找到
        required_shapes = ['W1', 'b1', 'W2', 'b2']
        if not all(shape in shapes for shape in required_shapes):
            print("Error: Missing some weight shapes in the file.")
            return None

        # 创建模型实例
        hidden_size = shapes['W1'][1]
        model = ThreeLayerNN(3072, hidden_size, 10, activation='relu')

        # 提取权重数据行
        weight_lines = [line for line in lines if line and not line.startswith(('W', 'b')) and 'shape' not in line and 'Best' not in line]

        if len(weight_lines) < 4:
            print("Error: Not enough weight data lines in file.")
            return None

        # 解析并设置权重
        def parse_weights(line, shape):
            try:
                values = [float(x) for x in line.split(',') if x.strip()]
                return np.array(values).reshape(shape)
            except ValueError as e:
                print(f"Error parsing weights: {e}")
                return None

        model.W1 = parse_weights(weight_lines[0], shapes['W1'])
        model.b1 = parse_weights(weight_lines[1], shapes['b1'])
        model.W2 = parse_weights(weight_lines[2], shapes['W2'])
        model.b2 = parse_weights(weight_lines[3], shapes['b2'])

        # 检查是否所有权重都成功加载
        if model.W1 is None or model.b1 is None or model.W2 is None or model.b2 is None:
            print("Error: Failed to load some weights.")
            return None

        print("Model weights loaded successfully!")
        return model

    except FileNotFoundError:
        print("Error: weight.txt file not found. Please run automatic parameter search first.")
        return None
    except Exception as e:
        print(f"Error loading weights: {str(e)}")
        return None

if __name__ == "__main__":
    # 加载数据
    data_dir = "data/cifar-10-batches-py"
    X_train, y_train, X_test, y_test = load_cifar10_data(data_dir)
    X_train, y_train, X_test, y_test = preprocess_data(X_train, y_train, X_test, y_test)

    # 划分验证集
    val_size = 1000
    X_val = X_train[:val_size]
    y_val = y_train[:val_size]
    X_train = X_train[val_size:]
    y_train = y_train[val_size:]

    print("CIFAR-10 Neural Network Training")
    print("1. Automatic parameter search")
    print("2. Manual parameter input")
    print("3. Load model from weight.txt")

    while True:
        try:
            choice = int(input("Please select mode (1, 2 or 3): "))
            if choice in [1, 2, 3]:
                break
            else:
                print("Please enter 1, 2 or 3")
        except ValueError:
            print("Please enter a number")

    if choice == 1:
        best_model, best_params = automatic_parameter_search(X_train, y_train, X_val, y_val)
    elif choice == 2:
        best_model, best_params = manual_parameter_input(X_train, y_train, X_val, y_val)
    else:
        best_model = load_model_from_weights()
        best_params = None

    # 在测试集上评估最佳模型
    if best_model is not None:
        test_acc = best_model.evaluate(X_test, y_test)
        print(f"\nTest Accuracy: {test_acc:.4f}")