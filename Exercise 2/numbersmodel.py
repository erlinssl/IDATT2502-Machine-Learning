import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt


mnist_train = torchvision.datasets.MNIST('./data', train=True, download=True)
x_train = mnist_train.data.reshape(-1, 784).float()  # Reshape input
y_train = torch.zeros((mnist_train.targets.shape[0], 10))  # Create output tensor
y_train[torch.arange(mnist_train.targets.shape[0]), mnist_train.targets] = 1  # Populate output

mnist_test = torchvision.datasets.MNIST('./data', train=False, download=True)
x_test = mnist_test.data.reshape(-1, 784).float()  # Reshape input
y_test = torch.zeros((mnist_test.targets.shape[0], 10))  # Create output tensor
y_test[torch.arange(mnist_test.targets.shape[0]), mnist_test.targets] = 1  # Populate output


def softmax(x):
    return torch.exp(x) / torch.sum(np.exp(x))


class NumbersModel:
    def __init__(self):
        self.W = torch.ones([784, 10], requires_grad=True)
        self.b = torch.ones([1, 10], requires_grad=True)

    def logits(self, x):
        return x @ self.W + self.b

    def f(self, x):
        return torch.nn.functional.softmax(self.logits(x))

    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy_with_logits(self.logits(x), y)

    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1), y.argmax(1)).float())


model = NumbersModel()

optimizer = torch.optim.SGD([model.W, model.b], 0.01)
for epoch in range(1500):
    print(epoch)
    model.loss(x_train, y_train).backward()
    optimizer.step()
    optimizer.zero_grad()

for index in range(0, 10):
    out = model.W[:, index].detach().numpy().reshape(28, 28)
    plt.imsave('./numbers/w_{num:d}.png'.format(num=index), out)

print("Loss = %s\nAccuracy = %s" % (model.loss(x_train, y_train), model.accuracy(x_test, y_test)))
