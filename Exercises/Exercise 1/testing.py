import torch
import matplotlib.pyplot as plt
import pandas as pd


x_train = torch.tensor([1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0]) # .reshape(-1, 1)  # x_train = [[1], [1.5], [2], [3], [4], [5], [6]]
print(x_train)


# x = x_values[:, 0]
# y = x_values[:, 1]
# z = y_values
#
# x_pred = np.linspace(0, 100, 30)
# y_pred = np.linspace(0, 100, 30)
# xx_pred, yy_pred = np.meshgrid(x_pred, y_pred)
# model_viz = np.array([xx_pred.flatten(), yy_pred.flatten()]).T
