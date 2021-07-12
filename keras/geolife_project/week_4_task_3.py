import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import seaborn as sns
import numpy as np

# This is the bottom left and top right edges
# of the picture that you are mapping latitude
# and longitude points to.
start_lat, end_lat = 39.7, 40.2
start_long, end_long = 116.0, 116.8

'''
This program creates a heatmap based on an image.

**CAUTION
You have to provide exact start_lat, end_lat, start_long, and end_long that correspond
to your bottom left snip of Google Maps photo and top right snip. This is a very first
version of the program, so the map is a little unaligned. Future versions will likely
user gmaps API.
'''
def main():
    
    # Path to map img
    img_path = 'D:/Documents/Research/UMBC/geolife_data/test'

    # Move to `final_dataframes`
    data_path = 'D:/Documents/Research/UMBC/geolife_data/Data2/final_dataframes'
    os.chdir(data_path)

    img = plt.imread(img_path + '/map.jpg')
    img_s = img.shape
    print(img_s)

    lat_range = [round(np.interp(i, [start_lat, end_lat], [0, img_s[1]]), 2) for i in np.arange(start_lat, end_lat, 0.05)]
    long_range = [round(np.interp(i, [start_long, end_long], [0, img_s[0]]), 2) for i in np.arange(start_long, end_long, 0.05)]

    lat_range_str = [str(i) for i in lat_range]
    long_range_str = [str(i) for i in long_range]

    # Base map
    plt.imshow(img, extent=(0, img_s[1], 0, img_s[0]))
    plt.title('Density map of trajectories captured by GPS - Beijing')
    plt.xlabel('latitude')
    plt.ylabel('longitude')
    plt.xticks(ticks=lat_range, labels=lat_range_str)
    plt.yticks(ticks=long_range, labels=long_range_str)

    # Start trajectories
    for u_id in range(182):

        user_traj_list = lat_long_pairs(u_id)

        plot_user_points(user_traj_list, img_s)

    plt.show()

'''
This function plots points into a pyplot plot for a particular user.
The parameter is a list of points in the following format
[((start_lat, start_long), (end_lat, end_long), 'timestamp')]
'''
def plot_user_points(traj_list, img_s):

    x_points = list()
    y_points = list()

    for traj in traj_list:
        if (traj[0][0] < 100) and (traj[0][1] < 500):
            x_points.append(traj[0][0])
            x_points.append(traj[1][0])
            y_points.append(traj[0][1])
            y_points.append(traj[1][1])
    
    x_points = [np.interp(i, [start_lat, end_lat], [0, img_s[1]]) for i in x_points]
    y_points = [np.interp(i, [start_long, end_long], [0, img_s[0]]) for i in y_points]

    plt.plot(x_points, y_points, 'bo', alpha=0.3, markersize=0.5)
    plt.plot(x_points, y_points, 'bo', alpha=0.0005, markersize=20)

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

if __name__ == '__main__':
    main()