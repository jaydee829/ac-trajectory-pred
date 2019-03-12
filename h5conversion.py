import numpy as np
import os
import h5py
import pandas as pd


def normalize(data):
    start_pos = [0, 0]
    columns = data.columns
    for column in columns:
        if 'attitude' in column:
            data = data.assign(
                **{'sin_' + column: lambda x: (np.sin(x[column])), 'cos_' + column: lambda x: (np.cos(x[column]))})
        elif any(x in column for x in ['vx', 'vy', 'vz']):
            data[column] /= 5000  # speeds < 50 m/s (5000cm/s) assumed
        elif any(x in column for x in ['lat', 'lon']):
            data[column] /= 1e7

            if 'lat' in column:
                start_pos[0] = data[column].iloc[0]
            elif 'lon' in column:
                start_pos[1] = data[column].iloc[0]

            position_normalize(data, column, start_pos)
            data[column].where(abs(data[column]) < 1., np.NaN, inplace=True)
            data[column].fillna(method='ffill', inplace=True)
        elif 'alt' in column:
            data[column] -= data[column].iloc[0]
            data[column] /= (1000 * 130)  # 130m ceiling assumed
            data[column].where(data[column].between(0, 1), np.NaN, inplace=True)
            data[column].fillna(method='ffill', inplace=True)
    return data


def position_normalize(data, label, start_pos):
    # range (-1,1) over 0.3 km either direction from start point
    dist_to_box = 0.3
    lat_box_ratio = dist_to_box / 110.574
    lon_box_ratio = dist_to_box / (111.320 * np.cos(start_pos[0] * 2 * np.pi / 360))

    if 'lat' in label:
        data[label] -= start_pos[0]
        data[label] /= lat_box_ratio
    elif 'lon' in label:
        data[label] -= start_pos[1]
        data[label] /= lon_box_ratio


def find_terminal_phases(data):
    takeoff_index = data[abs(data[' mavlink_global_position_int_t.vz']) > 50].index[0]
    landing_index = data[abs(data[' mavlink_global_position_int_t.vz']) > 50].index[-1]

    return data.iloc[takeoff_index:landing_index]



datapath = "./Data/"
all_files = os.listdir(datapath)
filelist = []
for file in all_files:
    if file.endswith('.csv') and file[0].isdigit():
        filelist.append(file)
output_filename = "./Data/data.h5"

val_set = np.random.randint(0, 4)
test_set = np.random.randint(0, 4)
while test_set == val_set:
    test_set = np.random.randint(0, 4)

with h5py.File(output_filename, 'w') as h5file:
    for file in filelist:
        data = pd.read_csv(datapath + file)
        drop_cols = ['Date', ' Time', ' mavlink_global_position_int_t.hdg', ' mavlink_gps_raw_int_t.time_usec',
                     ' mavlink_global_position_int_t.time_boot_ms']
        data.drop(drop_cols, axis=1, inplace=True)

        data = find_terminal_phases(data)

        stats = data.describe()

        data = normalize(data)

        drop_nonnormal_angles = [' mavlink_attitude_t.roll', ' mavlink_attitude_t.pitch', ' mavlink_attitude_t.yaw']

        data.drop(drop_nonnormal_angles, axis=1, inplace=True)

        if filelist[val_set] == file:
            flight_data = h5file.create_dataset(name=("flightdata/validation/" + file + "/data"), data=data.values)
            flight_stats = h5file.create_dataset(name=("flightdata/validation/" + file + "/stats"), data=stats.values)
        elif filelist[test_set] == file:
            flight_data = h5file.create_dataset(name=("flightdata/test/" + file + "/data"), data=data.values)
            flight_stats = h5file.create_dataset(name=("flightdata/test/" + file + "/stats"), data=stats.values)
        else:
            flight_data = h5file.create_dataset(name=("flightdata/train/" + file + "/data"), data=data.values)
            flight_stats = h5file.create_dataset(name=("flightdata/train/" + file + "/stats"), data=stats.values)

        flight_data.attrs['Column names'] = str(data.columns.values)
        flight_stats.attrs['Column names'] = str(stats.columns.values)
        flight_stats.attrs['Row names'] = str(stats.index.values)
