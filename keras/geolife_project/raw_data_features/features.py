import os
import pandas as pd
import numpy as np
import haversine as hs
from haversine import Unit
from time import time

'''
This function computes the Heading Change Rate
'''
def compute_hcr(df):
    
    df['HCR'] = 0.0
    df['HCR'][1:] = [(df['angle_diff'].values[i] / df['distance(meters)'].values[i]) for i in range(1, len(df))]

    return df

'''
This function computes the Velocity Change Rate
'''
def compute_vcr(df):

    df['VCR'] = 0.0
    df['VCR'][1:] = [((df['velocity(m/s)'].values[i] - df['velocity(m/s)'].values[i - 1]) / df['delta_time(seconds)'].values[i]) for i in range(1, len(df))]

    return df

'''
This function computes the angle difference. The units are still radians
'''
def compute_angle_diff(df):
    
    df['angle_diff'] = 0.0
    df['angle_diff'][1:] = [(df['bearing_angle(rad)'].values[i] - df['bearing_angle(rad)'].values[i - 1]) for i in range(1, len(df))]

    return df

'''
This function changes the velocity column. The units are m/s
'''
def compute_velocities(df):
    df['velocity(m/s)'] = 0.0

    df['velocity(m/s)'] = df['distance(meters)'] / df['delta_time(seconds)']

    return df

'''
Computes the bearing angles for the trajectories. The units are radians.
'''
def compute_bearing_angles(df):

    df['bearing_angle(rad)'] = 0.0

    # Iterate through the entire DataFrame
    for i in range(1, len(df)):

        if (i % 100000 == 0):
            print('step', i)

        # Get the first latitude and longitude point
        la1 = df['lat'].values[i - 1]
        lo1 = df['lon'].values[i - 1]

        # Get the second latitude and longitude point
        la2 = df['lat'].values[i]
        lo2 = df['lon'].values[i]

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
    
    df['delta_time(seconds)'] = 0.0

    # Iterate through the entire DataFrame
    for i in range(1, len(df)):

        if (i % 100000 == 0):
            print('step', i)

        # Get the first latitude and longitude point
        t1 = df['daytime'].values[i - 1]

        # Get the second latitude and longitude point
        t2 = df['daytime'].values[i]

        time1 = time_into_seconds(t1.split(' ')[1], t1.split(' ')[0])
        time2 = time_into_seconds(t2.split(' ')[1], t2.split(' ')[0])

        delta_time = time2 - time1

        df.at[i, 'delta_time(seconds)'] = delta_time

    return df

'''
Compute the distance column. The units are meters.
This returns a copy of the DataFrame with the distance
column completed.
'''
def compute_distance(df):

    df['distance(meters)'] = 0.0

    # Iterate through the entire DataFrame
    for i in range(1, len(df)):

        if (i % 100000 == 0):
            print('step', i)

        # Get the first latitude and longitude point
        la1 = df['lat'].values[i - 1]
        lo1 = df['lon'].values[i - 1]

        # Get the second latitude and longitude point
        la2 = df['lat'].values[i]
        lo2 = df['lon'].values[i]

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
        if dir != 'final_dataframes':
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

data_path = 'D:/Documents/Research/UMBC/geolife_data/Data2'
os.chdir(data_path)

print('Reading...')
df = pd.read_csv('./geolife_labeled.csv')

print('Printing...')
print(df.head(5))
print(df.tail(5))

# print('Computing delta_time...')
# s = time()
# df = compute_delta_time(df)
# e = time()
# print('Time taken:', e - s)

# print('Computing distance...')
# s = time()
# df = compute_distance(df)
# e = time()
# print('Time taken:', e - s)

# print('Computing velocities...')
# s = time()
# df = compute_velocities(df)
# e = time()
# print('Time taken:', e - s)

# print('Computing bearing angles...')
# s = time()
# df = compute_bearing_angles(df)
# e = time()
# print('Time taken:', e - s)

# print('Computing angle differences...')
# s = time()
# df = compute_angle_diff(df)
# e = time()
# print('Time taken:', e - s)

print('Computing HCR...')
s = time()
df = compute_hcr(df)
e = time()
print('Time taken:', e - s)

print('Computing VCR...')
s = time()
df = compute_vcr(df)
e = time()
print('Time taken:', e - s)

# After computing features
print('Printing...')
print(df.head(5))
inpt = input('Proceed?')
print('Saving...')
df.to_csv('./geolife_labeled.csv', index=False)
print('Printing...')
print(df.head(5))
print(df.tail(5))