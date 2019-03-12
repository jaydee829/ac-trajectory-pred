def my_rnn(input_shape):

    from keras import models, layers, optimizers

    model = models.Sequential()
    model.add(layers.LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(layers.LSTM(64, return_sequences=True))
    model.add(layers.LSTM(64, return_sequences=True))
    model.add(layers.LSTM(64, return_sequences=True))
    model.add(layers.TimeDistributed(layers.Dense(3)))

    model.compile(optimizer=optimizers.RMSprop(), loss='mse', metrics=['mse', 'mae'])

    return model


def rnn_1dconv_extraction(input_shape, params):

    from keras import models, layers, optimizers

    model = models.Sequential()
    model.add(layers.Conv1D(36, 6, activation='relu', input_shape=input_shape))
    model.add(layers.CuDNNLSTM(32, return_sequences=True))
    model.add(layers.CuDNNLSTM(32))
    model.add(layers.Dense(3))

    lr = params['lr']
    temp_params = params.copy()
    temp_params.pop('lr')

    model.compile(optimizer=optimizers.RMSprop(lr=lr), **temp_params)

    return model


def my_1dconv(input_shape):

    from keras import models, layers, optimizers

    model = models.Sequential()
    model.add(layers.Conv1D(32, 9, activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling1D(16))
    model.add(layers.Conv1D(32, 9, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(3))

    model.compile(optimizer=optimizers.RMSprop, loss='mse', metrics=['mse', 'mae'])


def rnn_dense_extraction(input_shape, params):

    from keras import models, layers, optimizers

    model = models.Sequential()
    model.add(layers.TimeDistributed(layers.Dense(64, activation='relu'), input_shape=input_shape))
    model.add(layers.CuDNNLSTM(64, return_sequences=True))
    model.add(layers.CuDNNLSTM(64, return_sequences=True))
    model.add(layers.TimeDistributed(layers.Dense(32, activation='relu')))
    model.add(layers.TimeDistributed(layers.Dense(3)))

    # model = models.Sequential()
    # model.add(layers.TimeDistributed(layers.Dense(64, activation='relu'), input_shape=input_shape))
    # model.add(layers.CuDNNLSTM(64, return_sequences=True))
    # model.add(layers.CuDNNLSTM(64, return_sequences=True))
    # model.add(layers.CuDNNLSTM(64, return_sequences=True))
    # model.add(layers.TimeDistributed(layers.Dense(64, activation='relu')))
    # model.add(layers.TimeDistributed(layers.Dense(32, activation='relu')))
    # model.add(layers.TimeDistributed(layers.Dense(3)))

    lr = params['lr']
    temp_params = params.copy()
    temp_params.pop('lr')

    model.compile(optimizer=optimizers.RMSprop(lr=lr), **temp_params)

    return model


def rnn_dense_extraction_notime(input_shape, params):

    from keras import models, layers, optimizers

    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=input_shape))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.CuDNNLSTM(128, return_sequences=True))
    model.add(layers.CuDNNLSTM(128, return_sequences=True))
    model.add(layers.CuDNNLSTM(64, return_sequences=True))
    model.add(layers.CuDNNLSTM(64, return_sequences=True))
    model.add(layers.CuDNNLSTM(64))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(3))

    lr = params['lr']
    temp_params = params.copy()
    temp_params.pop('lr')

    model.compile(optimizer=optimizers.RMSprop(lr=lr), **temp_params)

    return model


def multihead_rnn(input_shape, params):

    from keras import layers, Input, models, optimizers

    data_input = Input(shape=input_shape, name='input_data')
    x = layers.TimeDistributed(layers.Dense(64, activation='relu'))(data_input)
    x = layers.TimeDistributed(layers.Dense(64, activation='relu'))(x)
    x = layers.CuDNNLSTM(128, return_sequences=True)(x)
    x = layers.CuDNNLSTM(128, return_sequences=True)(x)
    x = layers.CuDNNLSTM(64, return_sequences=True)(x)

    lat_predict = layers.CuDNNLSTM(64, return_sequences=True)(x)
    lat_predict = layers.TimeDistributed(layers.Dense(64, activation='relu'))(lat_predict)
    lat_predict = layers.TimeDistributed(layers.Dense(32, activation='relu'))(lat_predict)
    lat_predict = layers.TimeDistributed(layers.Dense(1), name='lat_predict')(lat_predict)

    lon_predict = layers.CuDNNLSTM(64, return_sequences=True)(x)
    lon_predict = layers.TimeDistributed(layers.Dense(64, activation='relu'))(lon_predict)
    lon_predict = layers.TimeDistributed(layers.Dense(32, activation='relu'))(lon_predict)
    lon_predict = layers.TimeDistributed(layers.Dense(1), name='lon_predict')(lon_predict)

    alt_predict = layers.CuDNNLSTM(64, return_sequences=True)(x)
    alt_predict = layers.TimeDistributed(layers.Dense(64, activation='relu'))(alt_predict)
    alt_predict = layers.TimeDistributed(layers.Dense(32, activation='relu'))(alt_predict)
    alt_predict = layers.TimeDistributed(layers.Dense(1), name='alt_predict')(alt_predict)

    model = models.Model(data_input, [lat_predict, lon_predict, alt_predict])

    model.compile(optimizer=optimizers.RMSprop(lr=params['lr']), loss=[params['loss']]*3, metrics=params['metrics'])

    return model

# def rnn_1dconv_residuals(input_shape):



