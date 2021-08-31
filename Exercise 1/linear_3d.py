import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

x_values = pd.read_csv('day_length_weight.csv', usecols=['# day', 'length']).to_numpy()
y_values = pd.read_csv('day_length_weight.csv', usecols=['weight']).to_numpy()

x_train = torch.Tensor(x_values).reshape(-1, 2)
y_train = torch.Tensor(y_values).reshape(-1, 1)


class LinearRegressionModel:
    def __init__(self):
        self.W = torch.tensor([[0.0], [0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)

    def f(self, x):
        return torch.Tensor(x) @ self.W + self.b

    def loss(self, x, y):
        return torch.mean(torch.square(self.f(x) - y))


model = LinearRegressionModel()

epochs = 100000
step = 0.0000001

optimizer = torch.optim.SGD([model.W, model.b], step)
for epoch in range(epochs):
    model.loss(x_train, y_train).backward()
    optimizer.step()

    optimizer.zero_grad()

print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))

x_grid, y_grid = np.meshgrid(np.linspace(0, 2000, 100), np.linspace(30, 130, 25))
z_grid = np.empty([25, 100])

fig = plt.figure('Linear regression: 3D', figsize=(12, 4))
plot1 = fig.add_subplot(131, projection='3d')
plot2 = fig.add_subplot(132, projection='3d')
plot3 = fig.add_subplot(133, projection='3d')
plots = [plot1, plot2, plot3]

print("plottin")

for plot in plots:
    plot.plot_wireframe(np.array([[]]), np.array([[]]), np.array([[]]), color='green')
    plot.plot(x_train[:, 0].squeeze(), x_train[:, 1].squeeze(), y_train[:, 0].squeeze(),
               marker='o', color='blue', alpha=0.25)
    plot.set_xlabel('Age (Days)')
    plot.set_ylabel('Length')
    plot.set_zlabel('Weigth')
    plot.locator_params(nbins=4, axis='x')
    plot.locator_params(nbins=5, axis='x')

    for i in range(0, x_grid.shape[0]):
        for j in range(0, x_grid.shape[1]):
            z_grid[i, j] = model.f([[x_grid[i, j], y_grid[i, j]]])
    plot.plot_wireframe(x_grid, y_grid, z_grid, color='green', alpha=0.50)
# plot1.plot_wireframe()

plot1.view_init(elev=28, azim=120)
plot2.view_init(elev=4, azim=114)
plot3.view_init(elev=60, azim=165)

fig.suptitle('Loss = ' + str(model.loss(x_train, y_train).data) + ' , Epochs = '+str(epochs)+' , Step length = ' + str(step),
             fontsize=16)

plt.show()
