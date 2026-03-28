import numpy as np
import matplotlib.pyplot as plt

MSE = lambda y, y_pred: np.mean((y - y_pred) ** 2)
MAE = lambda y, y_pred: np.mean(np.abs(y - y_pred))
RMSE = lambda y, y_pred: np.sqrt(MSE(y, y_pred))
RMSLE = lambda y, y_pred: np.sqrt(np.mean((np.log1p(y) - np.log1p(y_pred)) ** 2))
R2_score = lambda y, y_pred: 1 - np.sum((y - y_pred) ** 2) / np.sum(
    (y - np.mean(y)) ** 2
)


def normalization(X):
    """
    paramter :
    X (numpy.ndarray): The input features.
    returns :
    X_norm (numpy.ndarray): The features after normalization.
    """
    x_min = np.min(X, axis=0)
    x_max = np.max(X, axis=0)
    X_range = x_max - x_min
    x_norm = (X - x_min) / (X_range + 10e-8)
    return x_norm


def plot_learning_curve(cost, label):
    plt.plot(cost, label=label)
    plt.xlabel("epochs")
    plt.ylabel("cost")
    plt.legend()
    plt.show()


def gd_compute_cost(y_true, y_pred):
    m = len(y_true)
    error = y_pred - y_true
    return (1 / (2 * m)) * np.dot(error.T, error)


def gradientDescent(X, y, learning_rate, epochs):
    """
    Performs gradient descent to optimize the weights of a linear regression model.
    Parameters:
    X (numpy.ndarray): The input features.
    y (numpy.ndarray): The target values.
    learning_rate (float): The learning rate for gradient descent.
    epochs (int): The number of iterations for gradient descent.
    Returns:
    numpy.ndarray: The optimized weights."""
    m = len(y)
    w_num = X.shape[1]
    W = np.zeros(w_num)
    cost_hestory = []
    for i in range(epochs):
        pred = np.dot(X, W)
        error = pred - y
        gradient = (1 / m) * np.dot(X.T, error)
        W = W - learning_rate * gradient
        cost_hestory.append(gd_compute_cost(y, pred))
    return W, np.array(cost_hestory)
