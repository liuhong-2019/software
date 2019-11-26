import random
import numpy as np

class FullConnectedLayer(object):
    def __init__(self, input_size, output_size, activator, learning_rate):
        self.input_size = input_size
        self.output_size = output_size
        self.activator = activator
        self.W = np.random.uniform(-0.1, 0.1, (output_size, input_size))
        self.b = np.zeros((output_size, 1))
        self.learning_rate = learning_rate
        self.output = np.zeros((output_size, 1))

    def forward(self, input_array):
        self.input = input_array
        self.output = self.activator.forward(
            np.dot(self.W, input_array) + self.b
        )


    def backward(self, delta_array):
        # self.delta = self.activator.backward(self.input) * np.dot(
        #     self.W.T, delta_array
        # )
        self.delta = np.multiply(
            self.activator.backward(self.input), np.dot(self.W.T, delta_array))
        self.W_grad = np.dot(delta_array, self.input.T)
        self.b_grad = delta_array

    def update(self):
        self.W  += self.learning_rate * self.W_grad
        self.b += self.learning_rate * self.b_grad

class SigmoidActivator(object):
    def forward(self, weighted_input):
        if weighted_input >= 0:
            return 1.0 / (1.0 + np.exp(-weighted_input))
        else:
            return np.exp(weighted_input)/(1 + np.exp(weighted_input))

    def backward(self, output):
        return output * (1 - output)

class Network(object):
    def __init__(self, layers):
        self.layers = []
        for i in range(len(layers) - 1):
            self.layers.append(
                FullConnectedLayer(layers[i], layers[i + 1], SigmoidActivator())
            )

    def predict(self, sample):
        sample = sample.reshape(-1, 1)
        output = sample
        for layer in self.layers:
            layer.forward(output)
            output = layer.output
        return output

    def train(self, labels, data_set, rate, epoch):
        # for d in range(len(data_set)):
        #     self.train_one_sample(labels[d], data_set[d], rate)
        for i in range(epoch):
            for d in range(len(data_set)):
                # print(i,'次迭代，',d,'个样本')
                oneobject = np.array(data_set[d]).reshape(-1, 1)  # 将输入对象和输出标签转化为列向量
                onelabel = np.array(labels[d]).reshape(-1, 1)
                self.train_one_sample(onelabel, oneobject, rate)


    def train_one_sample(self, label, sample, rate):
        self.predict(sample)
        self.calc_gradient(label)
        self.update_weight(rate)

    def calc_gradient(self, label):
        delta = self.layers[-1].activator.backward(
            self.layers[-1].output
        ) * (label - self.layers[-1].output)
        for layer in self.layers[::-1]:
            layer.backward(delta)
            delta = layer.delta
        return delta

    def update_weight(self, rate):
        for layer in self.layers:
            layer.update(rate)

def evaluate(network, test_data_set, test_labels):
    error = 0
    total = test_data_set.shape[0]
    for i in range(total):
        label = valye2type(test_labels[i])
        predict = valye2type(network.predict(test_data_set[i]))
        if label != predict:
            error += 1
    return float(error) / float(total)

def valye2type(vec):
    return vec.argmax(axis=0)

if __name__ == '__main__':
    # 使用神经网络实现and运算
    data_set = np.array([[0,0],[0,1],[1,0],[1,1]])
    labels = np.array([[1,0],[1,0],[1,0],[0,1]])
    # print(data_set)
    # print(labels)
    net = Network([2,1,2])  # 输入节点2个（偏量b会自动加上），神经元1个，输出节点2个。
    net.train(labels, data_set, 2, 100)
    for layer in net.layers:  # 网络层总不包含输出层
        print('W:',layer.W)
        print('b:',layer.b)

    # 对结果进行预测
    sample = np.array([[1,1]])
    result = net.predict(sample)
    type = valye2type(result)
    print('分类：',type)






