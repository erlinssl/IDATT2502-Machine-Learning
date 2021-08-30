import torch
import matplotlib.pyplot as plt
import csv


fields = []
rows = []

x_es = []
y_es = []

with open('length_weight.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)

    fields = next(csvreader)

    for row in csvreader:
        for col in row:
            x_es.append(float(row[0]))
            y_es.append(float(row[1]))

# x_values = pd.read_csv('length_weight.csv', usecols=['# length'])
# y_values = pd.read_csv('length_weight.csv', usecols=['weight'])

x_train = torch.tensor(x_es).reshape(-1, 1)
y_train = torch.tensor(y_es).reshape(-1, 1)

print(x_train)

class LinearRegressionModel:
    def __init__(self):
        self.W = torch.tensor([[0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)

    def f(self, x):
        return x @ self.W + self.b

    def loss(self, x, y):
        return torch.mean(torch.square(self.f(x) - y))


model = LinearRegressionModel()


optimizer = torch.optim.SGD([model.W, model.b], 0.0001)
for epoch in range(2500):
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
plt.show()
