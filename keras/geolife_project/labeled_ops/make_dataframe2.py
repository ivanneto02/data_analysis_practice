from datetime import time
from numpy.lib.function_base import msort
import pandas as pd
import numpy as np
from pandas.core import base
import os
import re

def main():

    pd.options.display.max_rows = 1000000

    data_path = 'D:/Documents/Research/UMBC/geolife_data/Data2/010/labels.txt'

    base_df = pd.read_csv(data_path, sep='\t')

    base_df.insert(0, 'user_id', '010')
    base_df.insert(1, 'trajectory_id', 0)
    base_df['Start_lat'] = 0
    base_df['End_lat'] = 0
    base_df['Start_long'] = 0
    base_df['End_long'] = 0
    base_df['label'] = base_df['Transportation Mode']
    base_df.drop(labels='Transportation Mode', inplace=True, axis=1)

    print(base_df)

    morph_df = pd.read_csv('D:/Documents/Research/UMBC/geolife_data/Data2/010/labelsOut.txt')

    print(morph_df)

    traj_df = pd.read_csv('D:/Documents/Research/UMBC/geolife_data/Data2/010/Trajectory/out.txt')

    dict = {}

    for i in range(0, len(morph_df)):
        dict[i] = ([morph_df['Start Time'][i], morph_df['End Time'][i]], [], [], morph_df['Transportation Mode'][i])

    for i in range(0, len(traj_df)):

        curr_time = traj_df['time_int'][i]
        found = False

        for j in range(0, len(dict)):
            
            if found:
                break
            
            if dict[j][0][0] <= curr_time <= dict[j][0][1]:
                dict[j][1].append(traj_df['lat'][i])
                dict[j][2].append(traj_df['long'][i])
                print('iteration:', i)

    print(len(dict))

    for i in range(0, len(base_df['Start_lat'])):
        
        if (len(dict[i][1]) != 0):
            base_df['trajectory_id'][i] = 1000000 + i
            base_df['Start_lat'][i] = dict[i][1][0]
            base_df['End_lat'][i] = dict[i][1][-1]
            base_df['Start_long'][i] = dict[i][2][0]
            base_df['End_long'][i] = dict[i][2][-1]

    print(base_df)

if __name__ == '__main__':
    main()