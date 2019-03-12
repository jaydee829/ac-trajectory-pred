import keras as K
import os
import yaml
import warnings


def get_model_number(root_path: str, params: dict):
    folders = list(filter(os.path.isdir, [root_path + s for s in os.listdir(root_path)]))
    recent = 0
    for folder in folders:
        if folder[len(root_path):len(root_path) + len('Model')] == 'Model':
            try:
                num = int(folder[-3:])
                recent = max(num, recent)
            except ValueError:
                warnings.warn('There are non-numbered Model folders present')

    params['model_num'] = recent + 1


def save_params(savepath: str, params: dict):
    try:
        os.makedirs(savepath)
    except FileExistsError:
        warnings.warn('Model number already exists')

    param_file = open(savepath + 'params.yaml', 'w')
    yaml.dump(params, param_file)
    param_file.close()


def save_run_info(root_path: str, params: dict):
    top_keys = ('modeltype', 'architecture', 'lr', 'start_epoch', 'epochs', 'steps_per_epoch', 'aaa')
    top_params = (params['modeltype'], params['basic_architecture'], params['compile_params']['lr'],
                  params['fit_params']['initial_epoch'], params['fit_params']['epochs'],
                  params['fit_params']['steps_per_epoch'], params['aaa'])
    summary = {'Model': params.get('model_num'), 'Params': dict(zip(top_keys, top_params))}
    try:
        save_file = open(root_path + 'Model_Guide.yaml', 'a')
        yaml.dump(summary, save_file)
        yaml.dump(''
                  '')
    except FileNotFoundError:
        save_file = open(root_path + 'Model_Guide.yaml', 'w')
        yaml.dump(summary, save_file)


import metrics
import generators

params = {}
fit_params = {}
compile_params = {}
params['fit_params'] = fit_params
params['compile_params'] = compile_params
fit_params['steps_per_epoch'] = 60
fit_params['epochs'] = 100
params['timesteps'] = 100
compile_params['lr'] = 0.001
compile_params['loss'] = 'mse'
compile_params['metrics'] = ['mae', metrics.lat_mse, metrics.lon_mse, metrics.alt_mse]
fit_params['initial_epoch'] = 0
params['modeltype'] = 'RNN_MLPExtraction'
params['aaa'] = 'changed lookahead to 3'

train_gen = generators.traj_generator('/Data/data.h5', fit_params.get('steps_per_epoch'), lookahead=3,
                                      purpose='train', timesteps=params.get('timesteps'), output=params['modeltype'])

val_gen = generators.traj_generator('/Data/data.h5', fit_params.get('steps_per_epoch'), lookahead=3,
                                    purpose='validation', timesteps=params.get('timesteps'),
                                    output=params['modeltype'])

import models

rootpath = '/Data/'

get_model_number(rootpath, params)

savepath = rootpath + 'Model_' + str(params.get('model_num')).zfill(3) + '/'

model = models.rnn_dense_extraction(input_shape=(None, 12), params=params.get('compile_params'))

reduce_lr = K.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=6, min_delta=0.01)
checkpoint = K.callbacks.ModelCheckpoint(savepath + 'model.{epoch:02d}-{val_loss:.2f}.hdf5', save_best_only=True)
logpath = savepath + 'history.csv'
logger = K.callbacks.CSVLogger(logpath, append=True)

fit_params['callbacks'] = [reduce_lr, checkpoint, logger]
params['log'] = logpath
params['num_layers'] = len(model.layers)
params['basic_architecture'] = [layer.name for layer in model.layers]

print(model.summary())

save_params(savepath, params)
save_run_info(rootpath, params)

history = model.fit_generator(train_gen, validation_data=val_gen, validation_steps=100, **fit_params)
