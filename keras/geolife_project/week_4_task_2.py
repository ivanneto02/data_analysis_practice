import os
import pandas as pd
import matplotlib.pyplot as plt
import re

'''
Only execute this program in the same directory as `final_df.csv`, OR adjust the data_path below
'''
def main():

    # Go into our main path with the dataframes
    data_path = 'D:/Documents/Research/UMBC/geolife_data/Data2/final_dataframes'
    os.chdir(data_path)

    # Which user to lookup
    user_input = input('Input the user to lookup (000-181): ')

    ## PROVIDE FUNCTION TO EXECUTE

    ##

    '''Example usage'''
    # Perform lookup
    user_list = list()

    for i in range(0, 182):
        for x in lat_long_pairs(i):
            user_list.append(x)

    # Create the histogram for `user_input`
    create_histograms(user_list, user_input)

'''
This function creates a histogram for the passed `u_id` user, based on a list in the format
[(start_lat, start_long), (end_lat, end_long), 'timestamp'].

x axis: hour in the day
y axis: how many trajectory starts there were in said hour
'''
def create_histograms(user_list, u_id):

    time_re = '(\d+[:]\d+[:]\d+)'

    x_range = list(range(24))
    x_range = [str(i) for i in x_range]

    time_list = list()
    height_list = list()

    for data in user_list:
        datetime = data[2]
        time = re.findall(time_re, datetime)[0][:2]
        print(time)
        time_list.append(int(time))
    
    for i in range(0, 24):

        curr_count = 0
        
        for time in time_list:
            if time == i:
                curr_count += 1

        height_list.append(curr_count)

    fig = plt.figure(figsize=(13, 7))
    plt.bar(x=x_range, height=height_list, alpha=0.5, ec='black', align='center', color='red')
    plt.title('Number of trajectory starts per hour in a day for user `{}`'.format(u_id))
    plt.xticks(x_range, color='black', fontsize='13')
    plt.xlabel('Hour (in a day)')
    plt.ylabel('Number of trajectory starts')
    plt.show()

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