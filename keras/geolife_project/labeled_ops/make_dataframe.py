from datetime import time
from numpy.lib.function_base import msort
import pandas as pd
import numpy as np
from pandas.core import base
import os
import re

def main():

    data_path = 'D:/Documents/Research/UMBC/geolife_data/Data2/010/labels.txt'

    base_df = pd.read_csv(data_path, sep='\t')

    base_df['Start_lat'] = 0
    base_df['End_lat'] = 0
    base_df['Start_long'] = 0
    base_df['End_long'] = 0
    base_df['label'] = 0

    print(base_df)

    morph_df = pd.read_csv('D:/Documents/Research/UMBC/geolife_data/Data2/010/labelsOut.txt')

    print(morph_df)

    traj_df = pd.read_csv('D:/Documents/Research/UMBC/geolife_data/Data2/010/Trajectory/out.txt')

    print(traj_df)

    print(traj_df['time_int'])


    print(min(traj_df['lat']))
    print(max(traj_df['lat']))
    print(min(traj_df['long']))
    print(max(traj_df['long']))

    # Iterate through every item in the time
    for i in range(0, len(morph_df['Start Time'])):
        
        lat_list = []
        long_list = []

        print('Iteration:', i)

        for j in range(0, len(traj_df['time_int'])):

            time_int = traj_df['time_int'][j]

            # print(time_int, '>=', morph_df['Start Time'][i])
            # print(' ', time_int, '<=', morph_df['End Time'][i])

            if (time_int >= morph_df['Start Time'][i]) and (time_int <= morph_df['End Time'][i]):
                lat_list.append(traj_df['lat'][j])
                long_list.append(traj_df['long'][j])
        
        if (len(lat_list) != 0):
            base_df['label'][i] = morph_df['Transportation Mode'][i]
            base_df['Start_lat'][i] = lat_list[0]
            base_df['End_lat'][i] = lat_list[len(lat_list) - 1]
            base_df['Start_long'][i] = long_list[0]
            base_df['End_long'][i] = long_list[len(long_list) - 1]

        print(base_df)

if __name__ == '__main__':
    main()