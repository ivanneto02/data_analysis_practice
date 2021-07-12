import pandas as pd
import os
import haversine as hs
from haversine import Unit
import numpy as np

'''
This program is meant to extract features from a huge dataset with
about 24 million rows. We will extract geospatial distance, temporal
distance, velocity, and bearing angle.
'''
def main():
    pd.options.display.max_rows = 10000

    data_path = 'D:/Documents/Research/UMBC/geolife_data/Data2'

    print('Opening raw DataFrame...')
    curr_df = pd.read_csv(data_path + '/uncompressed_df_label_10.csv')

    print('Dropping NaN...')
    curr_df.dropna(inplace=True, axis=0)

    print('Printing...')
    print(curr_df.head(10000))

'''
This function assigns each trajectory with a label. This label is obtained from another
DataFrame which contains ranges of time with a transportation mode.
'''
def find_labels(df_raw, df_range):

    df_raw['label'] = 'NaN'

    new_df = pd.DataFrame()

    i = 0
    j = 0
    while (i < len(df_raw)) and (j < len(df_range)):

        if (i % 100000 == 0):
            print('Step:', i)

        range_s = df_range['Start Time'].values[j].split(' ')
        range_e = df_range['End Time'].values[j].split(' ')

        range_s_sec = time_into_seconds(range_s[1], range_s[0])
        range_e_sec = time_into_seconds(range_e[1], range_e[0])

        # Find time at i
        raw_time = df_raw['datetime'].values[i].split(' ')
        raw_time_sec = time_into_seconds(raw_time[1], raw_time[0])

        if (range_s_sec <= raw_time_sec <= range_e_sec):

            df_raw.at[i, 'label'] = df_range['label'].values[j]
            i += 1
        elif (raw_time_sec < range_s_sec):
            i += 1
        else:
            j += 1

    print('Final...')
    print(df_raw[df_raw['label'] != 'unlabeled'].head(5))
    print('Appending...')
    new_df = new_df.append(df_raw)
    del df_raw
    del df_range

    return new_df

'''
This function computes the Heading Change Rate
'''
def compute_hcr(df):
    
    for i in range(1, len(df)):

        if (i % 100000 == 0):
            print('Step:', i)

        df.at[i, 'HCR'] = df['angle_diff'].values[i] / df['distance(meters)'].values[i]

    return df

'''
This function computes the Velocity Change Rate
'''
def compute_vcr(df):

    for i in range(1, len(df)):

        if (i % 100000 == 0):
            print('Step:', i)

        df.at[i, 'VCR'] = (df['velocity(m/s)'].values[i] - df['velocity(m/s)'].values[i - 1]) / df['delta_time(seconds)'].values[i]

    return df

'''
This function computes the angle difference. The units are still radians
'''
def compute_angle_diff(df):
    
    for i in range(1, len(df)):

        if (i % 100000 == 0):
            print('Step:', i)

        df.at[i, 'angle_diff'] = df['bearing_angle(rad)'].values[i] - df['bearing_angle(rad)'].values[i - 1]

    return df

'''
This function changes the velocity column. The units are m/s
'''
def compute_velocities(df):
    df['velocity(m/s)'] = pd.to_numeric(df['velocity(m/s)'], downcast='float')

    # Iterate through the entire DataFrame
    for i in range(1, len(df)):

        if (i % 100000 == 0):
            print('step', i)

        # Get the distance (meters)
        distance = df['distance(meters)'].values[i]

        # Get the delta_time (seconds)
        delta_time = df['delta_time(seconds)'].values[i]

        if delta_time == 0:
            df.at[i, 'velocity(m/s)'] = 'NaN'
            continue

        # Compute velocity (m/s)
        velocity = distance / delta_time

        df.at[i, 'velocity(m/s)'] = velocity

    return df

'''
Computes the bearing angles for the trajectories. The units are radians.
'''
def compute_bearing_angles(df):
    df['bearing_angle(rad)'] = pd.to_numeric(df['bearing_angle(rad)'], downcast='float')

    # Iterate through the entire DataFrame
    for i in range(1, len(df)):

        if (i % 100000 == 0):
            print('step', i)

        # Get the first latitude and longitude point
        la1 = df['lat'].values[i - 1]
        lo1 = df['long'].values[i - 1]

        # Get the second latitude and longitude point
        la2 = df['lat'].values[i]
        lo2 = df['long'].values[i]

        bearing_angle = compute_bearing_angle(la1, lo1, la2, lo2)

        df.at[i, 'bearing_angle(rad)'] = bearing_angle

    return df

'''
Computes the bearing angle. The bearing angle is the angle of a vector
in respect to the north (or unit vector (0, 1))
'''
def compute_bearing_angle(la1, lo1, la2, lo2):

    # Compute elements to pass to atan2
    X = np.math.sin(to_radians(lo2 - lo1)) * np.math.cos(to_radians(la2))
    # cos θa * sin θb – sin θa * cos θb * cos ∆L
    Y = np.math.cos(to_radians(la1)) * np.math.sin(to_radians(la2)) - np.math.sin(to_radians(la1)) * np.math.cos(to_radians(la2)) * np.math.cos(to_radians(lo2 - lo1))

    # Return the bearing angle
    return np.math.atan2(X, Y) + 2*np.math.pi

'''
This function converts the angle from degrees to radians
'''
def to_radians(angle):
    return angle * (np.math.pi / 180)

'''
Compute the delta_time column. The units are seconds.
This retuns a copy of the DataFrame with the delta_time
column completed.
'''
def compute_delta_time(df):
    df['delta_time(seconds)'] = pd.to_numeric(df['delta_time(seconds)'], downcast='float')
    
    # Iterate through the entire DataFrame
    for i in range(1, len(df)):

        if (i % 100000 == 0):
            print('step', i)

        # Get the first latitude and longitude point
        t1 = df['time'].values[i - 1]
        d1 = df['date'].values[i - 1]

        # Get the second latitude and longitude point
        t2 = df['time'].values[i]
        d2 = df['date'].values[i]

        time1 = time_into_seconds(t1, d1)
        time2 = time_into_seconds(t2, d2)

        delta_time = time2 - time1

        df.at[i, 'delta_time(seconds)'] = delta_time

    return df

'''
Compute the distance column. The units are meters.
This returns a copy of the DataFrame with the distance
column completed.
'''
def compute_distance(df):

    df['distance(meters)'] = pd.to_numeric(df['distance(meters)'], downcast='float')

    # Iterate through the entire DataFrame
    for i in range(1, len(df)):

        if (i % 100000 == 0):
            print('step', i)

        # Get the first latitude and longitude point
        la1 = df['lat'].values[i - 1]
        lo1 = df['long'].values[i - 1]

        # Get the second latitude and longitude point
        la2 = df['lat'].values[i]
        lo2 = df['long'].values[i]

        pt1 = (la1, lo1)
        pt2 = (la2, lo2)

        df.at[i, 'distance(meters)'] = hs.haversine(pt1, pt2, unit=Unit.METERS)

    return df

'''
This function takes in the DataPath where each user is in a
folder. Then, each users `out.txt` file (converged .PLT files)
will be merged into a huge pandas DataFrame.
'''
def merge_dataframe(dp):
    
    os.chdir(dp)

    my_df = pd.DataFrame()

    for dir in next(os.walk('.'))[1]:
        if (len(dir) == 3) and ('labels.txt' in next(os.walk('./' + dir))[2]):
            print('Merging out.txt for user {}...'.format(dir))
            
            os.chdir('./' + dir + '/Trajectory/')
                
            curr_df = pd.read_csv('./out.txt')
            my_df = my_df.append(curr_df)

            os.chdir('../../')

    return my_df

'''returns specific time in format 00:00:00 in seconds'''
def time_into_seconds(time, date):

    # Format time
    time_list = time.split(':')
    tot_time = 0
    for i in range(0, len(time_list)):

        rel_time = 0

        for j in range(0, 2):
            if j % 2 == 0:
                rel_time += int(time_list[i][j]) * 10
            else:
                rel_time += int(time_list[i][j])

        # Decreasing time weight (hours, minutes, seconds)
        if i % 3 == 0:
            tot_time += rel_time * 3600       
        elif i % 3 == 1:
            tot_time += rel_time * 60
        else:
            tot_time += rel_time * 1

    # Format date
    year = int(date[0:4]) * 31536000
    month = int(date[5:7]) * 2592000
    day = int(date[8:10]) * 86400

    tot_date = year + month + day

    return tot_time + tot_date

if __name__ == '__main__':
    main()