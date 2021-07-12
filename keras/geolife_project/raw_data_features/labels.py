import numpy as np
import pandas as pd
import glob
import os.path
import datetime
import os
from time import time

def read_plt(plt_file):
    points = pd.read_csv(plt_file, skiprows=6, header=None, parse_dates=[[5, 6]], infer_datetime_format=True)

    # for clarity rename columns
    points.rename(inplace=True, columns={'5_6':'daytime', 0: 'lat', 1: 'lon', 3: 'alt'})

    # remove unused columns
    points.drop(inplace=True, columns=[2, 4])

    return points

def read_labels(labels_file):
    labels = pd.read_csv(labels_file, skiprows=1, header=None,
                         parse_dates=[[0, 1], [2, 3]],
                         infer_datetime_format=True, delim_whitespace=True)

    # for clarity rename columns
    labels.columns = ['start_time', 'end_time', 'label']

    # replace 'label' column with integer encoding
    labels['label'] = [mode_ids[i] for i in labels['label']]

    return labels

def apply_labels(points, labels):
    indices = labels['start_time'].searchsorted(points['daytime'], side='right') - 1
    no_label = (indices < 0) | (points['daytime'].values >= labels['end_time'].iloc[indices].values)
    points['label'] = labels['label'].iloc[indices].values
    points['label'][no_label] = 0

def read_user(user_folder):
    labels = None

    plt_files = glob.glob(os.path.join(user_folder, 'Trajectory', '*.plt'))
    print('Number of trajectories found: {}'.format(len(plt_files)))
    df = pd.concat([read_plt(f) for f in sorted(plt_files)])

    labels_file = os.path.join(user_folder, 'labels.txt')
    if os.path.exists(labels_file):
        labels = read_labels(labels_file)
        apply_labels(df, labels)
    else:
        df['label'] = 0

    return df

def read_all_users(folder):
    subfolders = sorted(os.listdir(folder))
    print(subfolders)
    dfs = []
    for i, sf in enumerate(subfolders):
        if not sf.startswith('.') and len(sf) == 3:
            print('[%d/%d] processing user %s' % (i + 1, len(subfolders), sf))
            df = read_user(os.path.join(folder,sf))
            df['user'] = int(sf)
            dfs.append(df)
    return pd.concat(dfs)

data_path = 'D:/Documents/Research/UMBC/geolife_data'
os.chdir(data_path)

mode_names = ['walk', 'bike', 'bus', 'car', 'subway','train', 'airplane', 'boat', 'run', 'motorcycle', 'taxi']
mode_ids = {s : i + 1 for i, s in enumerate(mode_names)}

t0 = time()
df = read_all_users('Data2')
print('Time taken for processing:{}'.format(time() - t0))

df_labeled = df[df['label'] > 0]
df_labeled.to_csv('geolife_labeled.csv', index=False)
df = pd.read_csv('geolife_labeled.csv')
df.head()