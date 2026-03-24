import numpy as np

MSE = lambda y, y_pred: np.mean((y - y_pred) ** 2)
