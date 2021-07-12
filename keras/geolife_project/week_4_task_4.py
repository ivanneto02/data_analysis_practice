from datetime import time
import os
from matplotlib.pyplot import get
import pandas as pd
import numpy as np
import re
import haversine as hs
from haversine import Unit

'''
This program is meant for extracting features from our dataset. The dataset we constructed is
in the format [((start_lat, start_long), (end_lat, end_long), 'timestamp')]
'''
def main():

    pd.options.display.max_rows = 1000

    # Move to directory
    data_path = 'D:/Documents/Research/UMBC/geolife_data/Data2/final_dataframes'
    os.chdir(data_path)

    # Load Pandas DataFrame
    traj_df = pd.read_csv('./final_df.csv')

    # Add velocities and angles to the dataframe
    for i in range(len(traj_df)):

        # Get two points
        pt1 = (traj_df['Start_lat'].values[i], traj_df['Start_long'].values[i])
        pt2 = (traj_df['End_lat'].values[i], traj_df['End_long'].values[i])

        ts1 = traj_df['Start Time'].values[i]
        ts2 = traj_df['End Time'].values[i]

        if pt1[0] == 'NaN':
            traj_df.at[i, 'angle'] = 'NaN'
            traj_df.at[i, 'velocity'] = 'NaN'
            continue

        # Get angle
        theta = getAngle(pt1, pt2)

        # Get velocity
        vel = getVel(pt1, pt2, ts1, ts2)

        traj_df.at[i, 'angle'] = theta
        traj_df.at[i, 'velocity'] = vel

    traj_df.dropna(axis=0, inplace=True)

    # Add acceleration to the dataframe
    # **IMPORTANT: All NaN rows will be dropped, and the first point will not have a computed acceleration.
    for i in range(1, len(traj_df)):

        # Get four points
        v1 = traj_df['velocity'].values[i-1]
        v2 = traj_df['velocity'].values[i]

        # Get four timestamps
        ts1 = traj_df['Start Time'].values[i-1]
        ts2 = traj_df['Start Time'].values[i]
        
        # In the case we have a NaN point (we cannot compute acceleration)
        if pt1[0] == 'NaN':
            traj_df.at[i, 'acceleration'] = 'NaN'
            continue

        # Compute acceleration
        acc = getAcceleration(v1, v2, ts1, ts2)

        traj_df.at[i, 'acceleration'] = acc

    traj_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Rearrange the columns
    traj_df = traj_df[['user_id', 'trajectory_id', 'Start Time', 'End Time', 'Start_lat', 'End_lat', 'Start_long', 'End_long', 'angle', 'velocity', 'acceleration', 'label']]

    print(traj_df.loc[traj_df['velocity'] > 100])
    print(traj_df.shape)

    # traj_df.to_csv('./final_df_angle_vel.csv')

'''
Finds the acceleration in a set of 4 points
The format is:
v1: double (velocity)
v2: double (velocity)
ts1: time at v1
ts2: time at v2
Velocities are computed between ts1 and ts2, ts3 and ts4.
Acceleration is computed between ts1 and ts3, ts2 and ts4, then averaged together.
'''
def getAcceleration(v1, v2, ts1, ts2):

    # Regex for time
    t_re = '(\d{2}[:]\d{2}[:]\d{2})'
    # Regex for date
    d_re = '(\d{4}[-\/]\d{2}[-\/]\d{2})'

    # Make timestamps into seconds only
    ts1 = time_into_seconds(re.findall(t_re, ts1)[0], re.findall(d_re, ts1)[0])
    ts2 = time_into_seconds(re.findall(t_re, ts2)[0], re.findall(d_re, ts2)[0])

    # Get change in time to compute accelerations and then average
    dt = ts2 - ts1

    # Get the difference in velocities
    dv = v2 - v1

    # Return acceleration
    return dv/dt 

'''
Finds the velocity between two points, given two points and
two timestamps. Units are m/s
The format is:
p1: (x_1, y_1)
p2: (x_2, y_2)
ts1: time at p1
ts2: time at p2
'''
def getVel(pt1, pt2, ts1, ts2):

    # Regex for time
    t_re = '(\d{2}[:]\d{2}[:]\d{2})'
    # Regex for date
    d_re = '(\d{4}[-\/]\d{2}[-\/]\d{2})'

    t1 = re.findall(t_re, ts1)[0] # time1
    d1 = re.findall(d_re, ts1)[0] # date1

    t2 = re.findall(t_re, ts2)[0] # time2
    d2 = re.findall(d_re, ts2)[0] # date2

    # Make the time into seconds
    t1 = time_into_seconds(t1, d1)
    t2 = time_into_seconds(t2, d2)

    dt = t2 - t1 # Change in time
    # print(dt)
    dx = hs.haversine(pt1, pt2, unit=Unit.METERS)
    # print(dx)

    if dt == 0:
        return 'NaN'

    # Return dx/dt (units/second)
    return np.divide(dx, dt)

'''
Finds the angle between two points, with basis being North
The format is:
p1: (x_1, y_1)
p2: (x_2, y_2)
'''
def getAngle(pt1, pt2):

    # la1 = pt1[0] * (np.math.pi / 180)
    # la2 = pt2[0] * (np.math.pi / 180)
    # lo1 = pt1[1] * (np.math.pi / 180)
    # lo2 = pt2[1] * (np.math.pi / 180)

    # X = np.math.cos() * np.math.sin() - np.math.sin() * np.math.cos() * np.math.cos() 
    # Y = 

    # Vectors
    north = [0, 1]
    vector = [pt2[0] - pt1[0], pt2[1] - pt1[1]]

    # Make unit vector
    if (np.linalg.norm(vector) == 0):
        return 'NaN'

    north_unit = np.divide(north, np.linalg.norm(north))
    vector_unit = np.divide(vector, np.linalg.norm(vector))

    # Find the dot product
    dot_p = np.dot(north_unit, vector_unit)

    # Return the angle
    return np.arccos(dot_p)

'''
Stop Rate

Rate at which the trajectory stops
'''
def extract_SR():
    pass

'''
Heading Change Rate

Rate at which the trajectory changes heading
'''
def extract_HCR():
    pass

'''
Velocity Change Rate

Rate at which the velocity changes
'''
def extract_VCR():
    pass

'''
Looks up a user in our DataFrame
Returns a lsit of latitude and longitude pairs

**IMPORTANT
This function is dropping NaN values from the DataFrame, meaning we are technically losing
information.
'''
def lat_long_pairs(u_id):

    my_df = pd.read_csv('./final_df.csv')

    my_list = list()

    des_df = my_df[my_df['user_id'] == int(u_id)].dropna(axis=0)

    for i in range(1, len(des_df)):

        start_lat = des_df['Start_lat'].iloc[i]
        end_lat = des_df['End_lat'].iloc[i]
        start_long = des_df['Start_long'].iloc[i]
        end_long = des_df['End_long'].iloc[i]
        timestamp = des_df['Start Time'].iloc[i]

        my_list.append(((start_lat, start_long), (end_lat, end_long), (timestamp)))

    return my_list

'''
Returns specific time in seconds
The format is:
time = '00:00:00'
date = 'yyyy/mm/dd' OR 'yyyy-mm-dd'
'''
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