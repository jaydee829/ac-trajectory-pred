import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

matplotlib.use('TkAGG')


def loss_plot(history):
    plt.figure()
    epochs = range(1, len(history.loss) + 1)
    plt.plot(epochs, history.loss, 'bo', label='Training loss')
    plt.plot(epochs, history.val_loss, 'b', label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()


def mse_plot(history):
    plt.figure()
    epochs = range(1, len(history.loss) + 1)
    plt.plot(epochs, history.mean_squared_error, 'bo', label='Training MSE')
    plt.plot(epochs, history.val_mean_squared_error, 'b', label='Validation MSE')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.show()


def load_history(datapath, models):
    dirlist = ['Model_' + str(model).zfill(3) + '/' for model in models]
    history = []
    for i, dir in enumerate(dirlist):
        history.append(pd.read_csv(datapath + dir + 'history.csv'))
    return history


def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def plot_training():
    models = [28]
    histories = load_history('/Data/', models)

    fig, axs = plt.subplots(1, 1)

    for model, history in zip(models, histories):
        epochs = range(1, len(history.loss) + 1)
        axs.plot(epochs, history.loss, marker='o', label='Model ' + str(model) + ' Training loss')
        axs.plot(epochs, history.val_loss, label='Model ' + str(model) + ' Validation loss')
        # axs.plot(epochs, smooth(history.loss, 5), marker='o', label='Model ' + str(model) + 'Smoothed Training loss')
        # axs.plot(epochs, smooth(history.val_loss, 5), label='Model ' + str(model) + 'Smoothed Validation loss')
        axs.set_xlabel('Epochs')
        axs.set_ylabel('Mean Squared Error')
        axs.set_ylim(top=0.1)

        plt.legend()

        # axs[1].plot(epochs, history.mean_squared_error, marker='o', label='Model '+model+'Training MSE')
        # axs[1].plot(epochs, history.val_mean_squared_error, label='Model '+model+'Validation MSE')
        # axs[1].set_xlabel('Epochs')
        # axs[1].set_ylabel('Mean Squared Error')

    # plt.show()
    plt.savefig('/Data/training_hist'+str(models)+'.png', format='png')


def plot_model():
    from keras.utils.vis_utils import plot_model
    from keras import models
    import metrics

    model_num = 20

    filepath = '/Data/Model_' + str(model_num).zfill(3) + '/'

    files = os.listdir(filepath)
    files = [file for file in files if 'model.' in file]

    model = models.load_model(filepath + files[0],
                              custom_objects={'lat_mse': metrics.lat_mse, 'lon_mse': metrics.lon_mse,
                                              'alt_mse': metrics.alt_mse})

    plot_model(model, to_file=filepath+'model_topology')


plot_training()
