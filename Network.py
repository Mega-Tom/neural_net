import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def bias(array):
    array = np.array(array)
    if len(array.shape) == 1:
        return np.append(array, [1])
    return np.vstack((array, np.ones((1, array.shape[1]))))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Network:
    def __init__(self, layers, learning_rate=0.1, mass=0):
        self.weights = []
        self.delta = []
        last_layer = None
        for layer in layers:
            if last_layer != None:
                self.weights.append(np.random.rand(layer, last_layer + 1) - 0.5)
                self.delta.append(np.zeros((layer, last_layer + 1)))
            last_layer = layer
        self.eta = learning_rate
        self.mass = mass
        
    def forward(self, inputs):
        data = inputs
        for layer in self.weights:
            data = sigmoid(layer @ bias(data))
        return data
    
    def back(self, inputs, targets):
        data = [inputs]
        for layer in self.weights:
            data.append(sigmoid(layer @ bias(data[-1])))
        error = [x * 0 for x in data]
        error[-1] = -(targets - data[-1]) * (data[-1]) * (1 - data[-1])
        for i in range(len(data) - 2, -1, -1):
            error[i] = (np.transpose(self.weights[i]) @ error[i + 1])
            error[i] = np.delete(error[i], -1)
            error[i] *= (data[i]) * (1 - data[i])
            self.weights[i] -= np.outer(error[i + 1], bias(data[i])) * self.eta
    
    def back_batch(self, inputs, targets):
        data = [np.transpose(inputs)]
        for layer in self.weights:
            data.append(sigmoid(layer @ bias(data[-1])))
        error = [0 for x in data]
        error[-1] = -(np.transpose(targets) - data[-1]) * (data[-1]) * (1 - data[-1])
        for i in range(len(data) - 2, -1, -1):
            error[i] = (np.transpose(self.weights[i]) @ error[i + 1])
            error[i] = np.delete(error[i], -1, axis=0)
            error[i] *= (data[i]) * (1 - data[i])
            self.delta[i] *= self.mass
            self.delta[i] += (error[i + 1] @ np.transpose(bias(data[i]))) * self.eta
            self.weights[i] -= self.delta[i]

def oneHot(data, classes):
    return np.array([[(1 if c == n else 0) for n in range(classes)] for c in data])

n = Network([4,2,3], 0.00869, 0.5459)

iris = datasets.load_iris()

data = iris.data
targ = iris.target

train_x, test_x, train_y, test_y = train_test_split(data, targ,
                                            train_size=0.7, test_size=0.3,
                                            random_state=0)
a = []
for _ in range(500):
   n.back_batch(train_x, oneHot(train_y, 3))
   a.append(np.mean([np.argmax(n.forward(test_x[i])) == test_y[i] for i in range(0, len(test_x))])*100)

print(np.mean([np.argmax(n.forward(test_x[i])) == test_y[i] for i in range(0, len(test_x))])*100)
plt.plot(a)
plt.ylabel('% accuracy')
plt.show()