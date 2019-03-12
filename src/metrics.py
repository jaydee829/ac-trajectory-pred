import numpy as np
import keras.backend as K


def lat_mse(y_true, y_pred):
    idx = 0
    return K.mean(np.square(y_true[:, idx] - y_pred[:, idx]))


def lon_mse(y_true, y_pred):
    idx = 1
    return K.mean(np.square(y_true[:, idx] - y_pred[:, idx]))


def alt_mse(y_true, y_pred):
    idx = 2
    return K.mean(np.square(y_true[:, idx] - y_pred[:, idx]))
