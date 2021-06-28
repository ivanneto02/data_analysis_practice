import os
import datetime, time
import re

def main():

    # Path to the `Data` directory
    data_path = 'D:/Documents/Research/UMBC/geolife_data/Data2'

    # Change to the data dir
    os.chdir(data_path)

    # For every dir in data dir
    for dir in next(os.walk('.'))[1]:
        print(dir)
        os.chdir(dir)

        # Only check the ones that have `labels.txt`
        if 'labels.txt' in next(os.walk('.'))[2]:
            
            # Move into ./Trajectory dir
            os.chdir('./Trajectory')
            
            fout = open('out.txt', 'wt')

            # Print files in this dir
            for file in next(os.walk('.'))[2]:

                with open(file, 'rt') as file1:
                    
                    for line in file1:
                        # Regular expr: '(\d+[:]\d+[:]\d+)'
                        match_time_list = re.findall('(\d+[:]\d+[:]\d+)', line)
                        match_date_list = re.findall('(\d{4}[-]\d{2}[-]\d{2})', line)

                        if (len(match_time_list) > 0):

                            match_time = match_time_list[0]
                            match_date = match_date_list[0]

                            fout.write(line.replace(match_date + "," + match_time, str(time_into_seconds(match_time, match_date))))
                    print(file, '-', dir)

            os.chdir('..')

        os.chdir('..')

    print('Done!!')

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

'''Used to find the number of lines in a file - UNUSED'''
# def get_line_count(file):

#     line_count = 0

#     for line in file:
#         if line != '\n':
#             line_count += 1

#     return line_count

def make_data_frame(file):
    pass

if __name__ == '__main__':
    main()