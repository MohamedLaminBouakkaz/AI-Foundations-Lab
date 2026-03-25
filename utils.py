import numpy as np
import matplotlib.pyplot as plt

MSE = lambda y, y_pred: np.mean((y - y_pred) ** 2)


def plot_learning_curve(cost, label):
    plt.plot(cost, label=label)
    plt.xlabel("epochs")
    plt.ylabel("cost")
    plt.legend()
    plt.show()
