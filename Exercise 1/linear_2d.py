import torch
import matplotlib.pyplot as plt
import csv
import pandas as pd


# x_es = []
# y_es = []
# with open('length_weight.csv', 'r') as csvfile:
#     csvreader = csv.reader(csvfile)
#     next(csvreader)
#     for row in csvreader:
#         for col in row:
#             x_es.append(float(row[0]))
#             y_es.append(float(row[1]))
#
# x_train = torch.tensor(x_es).reshape(-1, 1)
# y_train = torch.tensor(y_es).reshape(-1, 1)

x_values = pd.read_csv('length_weight.csv', usecols=['# length']).to_numpy()
y_values = pd.read_csv('length_weight.csv', usecols=['weight']).to_numpy()
x_train = torch.Tensor(x_values).reshape(-1, 1)
y_train = torch.Tensor(y_values).reshape(-1, 1)


class LinearRegressionModel:
    def __init__(self):
        self.W = torch.tensor([[0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)

    def f(self, x):
        return x @ self.W + self.b

    def loss(self, x, y):
        return torch.mean(torch.square(self.f(x) - y))


model = LinearRegressionModel()


optimizer = torch.optim.SGD([model.W, model.b], 0.000000001)
for epoch in range(10000):
    model.loss(x_train, y_train).backward()
    optimizer.step()

    optimizer.zero_grad()

print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))

plt.plot(x_train, y_train, 'o', label='$(x^{(i)},y^{(i)})$')
plt.xlabel('x')
plt.ylabel('y')
x = torch.tensor([[torch.min(x_train)], [torch.max(x_train)]])  # x = [[1], [6]]]
plt.plot(x, model.f(x).detach(), label='$\\hat y = f(x) = xW+b$')
plt.legend()
plt.text(42.75, 17, 'Loss = ' + str(model.loss(x_train, y_train).data))
plt.show()
