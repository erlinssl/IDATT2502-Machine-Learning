import numpy as np
import matplotlib.pyplot as plt
import torch

x_train = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y_train = torch.tensor([[0.], [1.], [1.], [.0]])


def sigmoid_function(z):
    return 1 / (1 + np.e ** -z)


class XorModel:
    def __init__(self):
        self.W = torch.tensor([[0.35, -0.21], [0.35, 0.65]], requires_grad=True)
        self.b = torch.tensor([[-0.65, -0.93]], requires_grad=True)
        self.W_2 = torch.tensor([[0.11], [-0.71]], requires_grad=True)
        self.b_2 = torch.tensor([[0.93]], requires_grad=True)

    def f_1(self, x):
        return sigmoid_function(x @ self.W + self.b)

    def f_2(self, h):
        return sigmoid_function(h @ self.W_2 + self.b_2)

    def f(self, x):
        return self.f_2(self.f_1(torch.Tensor(x)))

    def loss(self, x, y):
        return -torch.mean(y * torch.log(self.f(x)) + (1 - y) * torch.log(1 - self.f(x)))


model = XorModel()
optimizer = torch.optim.SGD([model.W, model.W_2, model.b, model.b_2], 1)
for epoch in range(75000):
    model.loss(x_train, y_train).backward()
    optimizer.step()
    optimizer.zero_grad()

fig = plt.figure('XOR operator')

plot = fig.add_subplot(111, projection='3d')

plot.set_xlabel('$x_1$')
plot.set_ylabel('$x_2$')
plot.set_zlabel('y')
plot.set_xticks([0, 1])
plot.set_yticks([0, 1])
plot.set_zticks([0, 1])
plot.set_xlim(-0.25, 1.25)
plot.set_ylim(-0.25, 1.25)
plot.set_zlim(-0.25, 1.25)

x1_grid, x2_grid = np.meshgrid(np.linspace(-0.25, 1.25, 10), np.linspace(-0.25, 1.25, 10))
y_grid = np.empty([10, 10])

plot.plot(x_train[:, 0].squeeze(), x_train[:, 1].squeeze(), y_train[:, 0].squeeze(),
          'o', color='blue', label='$(x_1^{(i)},x_2^{(i)},y^{(i)})$')
fig.suptitle('XOR operator')

plot_f = plot.plot_wireframe(np.array([[]]), np.array([[]]), np.array([[]]), color="green",
                             label="$\\hat y=f(\\mathbf{x})=\\sigma(\\mathbf{xW}+b)$")
for i in range(0, x1_grid.shape[0]):
    for j in range(0, x1_grid.shape[1]):
        y_grid[i, j] = model.f([[x1_grid[i, j], x2_grid[i, j]]])
plot.plot_wireframe(x1_grid, x2_grid, y_grid, color='green', alpha=0.75)

table = plt.table(cellText=[[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]],
                  colWidths=[0.1] * 3,
                  colLabels=["$x_1$", "$x_2$", "$f(\\mathbf{x})$"],
                  cellLoc="center",
                  loc="upper right")

plot_info = fig.text(0.01, 0.02, "$W_1$ = %s\n$b_1$ = %s\n$W_2$ = %s\n $b_2$ = %s\n"
                                 "loss = %f" % (model.W.data, model.b, model.W_2, model.b_2, model.loss(x_train, y_train)))

plot.legend(loc='upper left')

plt.show()
