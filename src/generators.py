import numpy as np
import h5py


def traj_generator(data_path: str, n_batches: int, timesteps: int = 500, lookahead: int = 3, purpose: str = 'train',
                   output: str = None):
    with h5py.File(name=data_path, mode='r') as file:
        flights = list(file['/flightdata/' + purpose].keys())
        flight_choice = np.random.choice(flights, n_batches)
        samples = np.empty([n_batches, timesteps, 12])
        targets = np.empty([n_batches, timesteps, 3])
        while True:
            for idx, choice in enumerate(flight_choice):
                data_to_slice = file['/flightdata/' + purpose + '/' + choice + '/data']
                start_index = np.random.randint(0, data_to_slice.len() - timesteps - lookahead)
                samples[idx] = data_to_slice[start_index:start_index + timesteps]
                targets[idx] = data_to_slice[start_index + lookahead:start_index + timesteps + lookahead, 0:3]
            if 'multi' in output.lower():
                yield {'input_data': samples}, {'lat_predict': np.expand_dims(targets[:, :, 0], axis=2),
                                                'lon_predict': np.expand_dims(targets[:, :, 1], axis=2),
                                                'alt_predict': np.expand_dims(targets[:, :, 2], axis=2)}
            else:
                yield samples, targets


def single_return_gen(data_path: str, n_batches: int, timesteps: int = 12, lookahead: int = 3, purpose: str = 'train',
                      output: str = None):
    with h5py.File(name=data_path, mode='r') as file:
        flights = list(file['/flightdata/' + purpose].keys())
        flight_choice = np.random.choice(flights, n_batches)
        samples = np.empty([n_batches, timesteps, 12])
        targets = np.empty([n_batches, timesteps, 3])
        while True:
            for idx, choice in enumerate(flight_choice):
                data_to_slice = file['/flightdata/' + purpose + '/' + choice + '/data']
                start_index = np.random.randint(0, data_to_slice.len() - timesteps - lookahead)
                samples[idx] = data_to_slice[start_index:start_index + timesteps]
                targets[idx] = data_to_slice[start_index + timesteps + lookahead, 0:3]
            if 'multi' in output.lower():
                yield {'input_data': samples}, {'lat_predict': np.expand_dims(targets[:, 0], axis=1),
                                                'lon_predict': np.expand_dims(targets[:, 1], axis=1),
                                                'alt_predict': np.expand_dims(targets[:, 2], axis=1)}
            else:
                yield samples, targets


def sequence_generator(data_path: str, n_batches: int, timesteps: int = 1, lookahead: int = 3,
                       purpose: str = 'validation'):
    with h5py.File(name=data_path, mode='r') as file:
        flights = list(file['/flightdata/' + purpose].keys())
        flight_choice = np.random.choice(flights, 1)
        samples = np.empty([1, timesteps, 12])
        targets = np.empty([1, timesteps, 3])
        start_index = 0
        while True:
            data_to_slice = file['/flightdata/' + purpose + '/' + flight_choice[0] + '/data']
            try:
                samples = data_to_slice[start_index:start_index + timesteps]
                targets = data_to_slice[start_index + lookahead:start_index + timesteps + lookahead, 0:3]
            except IndexError:
                flight_choice = np.random.choice(flights, 1)
                start_index = 0
                continue
            start_index += 1
            yield np.expand_dims(samples, axis=0), np.expand_dims(targets, axis=0)
