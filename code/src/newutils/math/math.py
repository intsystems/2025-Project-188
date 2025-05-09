import numpy as np


def pseudo_mape_obj(preds, dtrain, delta=1e-1, eps=1e-9):
    delta_squared = delta * delta

    y = dtrain.get_label()
    y_safe = np.where(y == 0, eps, y)

    r = (preds - y) / y_safe
    denom = np.sqrt(r * r + delta_squared)
    grad = (r / denom) / y_safe
    hess = (delta_squared / (denom**3)) / (y_safe * y_safe)

    return grad, hess
