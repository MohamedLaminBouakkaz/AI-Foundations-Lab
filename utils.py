import numpy as np
import matplotlib.pyplot as plt

MSE = lambda y, y_pred: np.mean((y - y_pred) ** 2)
MAE = lambda y, y_pred: np.mean(np.abs(y - y_pred))


def plot_learning_curve(cost, label):
    plt.plot(cost, label=label)
    plt.xlabel("epochs")
    plt.ylabel("cost")
    plt.legend()
    plt.show()


def gradientDescent(X, y, learning_rate, epochs):
    """
    Performs gradient descent to optimize the weights of a linear regression model.
    Parameters:
    X (numpy.ndarray): The input features.
    y (numpy.ndarray): The target values.
    learning_rate (float): The learning rate for gradient descent.
    epochs (int): The number of iterations for gradient descent.
    Returns:
    numpy.ndarray: The optimized weights.
    numpy.ndarray: The cost values for each epoch."""
    m = len(y)
    X = np.c_[X, np.ones(m)]
    w_num = X.shape[1]
    W = np.zeros(w_num)
    for i in range(epochs):
        pred = np.dot(X, W)
        error = pred - y
        gradient = (1 / m) * np.dot(X.T, error)
        W = W - learning_rate * gradient
    return W
