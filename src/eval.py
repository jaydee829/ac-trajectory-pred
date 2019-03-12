import numpy as np
import matplotlib
import os
import matplotlib.pyplot as plt
from keras import models
import metrics
import generators
import yaml
import scipy.stats as st

matplotlib.use('TkAGG')


def flight_eval(model_nums: list, purpose: str = 'validation'):
    lookahead = 3
    seq_gen = generators.sequence_generator('/Data/data.h5', n_batches=11, timesteps=100, lookahead=lookahead,
                                            purpose=purpose)
    custom_objects = {'lat_mse': metrics.lat_mse, 'lon_mse': metrics.lon_mse, 'alt_mse': metrics.alt_mse}

    for model_num in model_nums:
        filepath = '/Data/Model_' + str(model_num).zfill(3) + '/'

        files = os.listdir(filepath)
        files = [file for file in files if 'model.' in file]

        model = models.load_model(filepath + max(files), custom_objects=custom_objects)

        eval_hist = model.evaluate_generator(seq_gen, steps=11)

        eval_hist = [val.item() for val in eval_hist]

        eval_dict = dict(zip(model.metrics_names, eval_hist))
        eval_file = open(filepath + purpose + '_eval'+str(lookahead)+'.yaml', 'w')
        yaml.dump(eval_dict, eval_file)
        eval_file.close()


def predict_eval(model_nums: list, purpose: str = 'test'):
    lookahead = 3
    seq_gen = generators.sequence_generator('/Data/data.h5', n_batches=11, timesteps=1, lookahead=lookahead,
                                            purpose=purpose)
    custom_objects = {'lat_mse': metrics.lat_mse, 'lon_mse': metrics.lon_mse, 'alt_mse': metrics.alt_mse}

    for model_num in model_nums:
        filepath = '/Data/Model_' + str(model_num).zfill(3) + '/'

        files = os.listdir(filepath)
        files = [file for file in files if 'model.' in file]

        model = models.load_model(filepath + max(files), custom_objects=custom_objects)

        predict_vals = model.predict_generator(seq_gen, steps=1975)
        np.squeeze(predict_vals)

        errors = np.empty([1975, 3])
        error_seq_gen = generators.sequence_generator('/Data/data.h5', n_batches=11, timesteps=1, lookahead=lookahead,
                                                      purpose=purpose)

        for idx, val in enumerate(predict_vals):
            _, truth = next(error_seq_gen)
            errors[idx] = np.multiply((val - truth), [300, 300, 130])

        t = st.t.ppf(0.025, errors.shape[0])
        eval_file = open(filepath + purpose + '_eval'+str(lookahead)+'.yaml', 'a')
        names = ['lat', 'lon', 'alt']
        for i, error in enumerate(np.rollaxis(errors,1)):
            stat_nums = [error.mean(), error.std(), error.mean() - t * error.std(), error.mean() + t * error.std()]
            stat_nums = [val.item() for val in stat_nums]
            stats = ['mean_error', 'mean_std', '0.95_lo', '0.95_hi']
            yaml.dump({names[i]: dict(zip(stats, stat_nums))}, eval_file)
        eval_file.close()

        x = np.linspace(0, errors.shape[0] / 3, errors.shape[0])
        fig, axs = plt.subplots(3, 1, figsize=(6, 6))
        axs[0].plot(x, errors[:, 0], 'b')
        axs[0].set_ylabel('Latitude Error')
        axs[0].set_ylim([-250, 250])
        axs[0].set_xlim(left=0)
        axs[0].axhline(color='k')
        axs[0].grid(True, linestyle=':')
        axs[1].plot(x, errors[:, 1], 'b')
        axs[1].set_ylabel('Longitude Error')
        axs[1].set_ylim([-250, 250])
        axs[1].set_xlim(left=0)
        axs[1].axhline(color='k')
        axs[1].grid(True, linestyle=':')
        axs[2].plot(x, errors[:, 2], 'b')
        axs[2].set_ylabel('Altitude Error')
        axs[2].set_xlabel('Time (s)')
        axs[2].set_ylim([-100, 50])
        axs[2].set_xlim(left=0)
        axs[2].axhline(color='k')
        axs[2].grid(True, linestyle=':')
        plt.savefig(filepath + 'test_error.png', format='png')


flight_eval([28], purpose='test')
predict_eval([28], purpose='test')
