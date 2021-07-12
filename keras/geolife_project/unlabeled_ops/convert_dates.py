import os
import re

def main():
    '''
    The aim of this file is to convert all of the dates inside out.txt into a single int with the number
    of seconds that the date represents
    '''

    data_path = 'D:/Documents/Research/UMBC/geolife_data/Data2'

    os.chdir(data_path)

    for dir in next(os.walk('.'))[1]:

        os.chdir('./' + dir)

        if ('labels.txt' not in next(os.walk('.'))[2]) and (dir != 'final_dataframes'):

            # Access `Trajectory` folder
            os.chdir('./Trajectory/')

            fout = open('out_int_dates.txt', 'wt')
            fout.write('lat,long,?1,?2,alt,time_int\n')

            # Regex for date item (\d{4}[-]\d{2}[-]\d{2})
            date_re = '(\d{4}[-]\d{2}[-]\d{2})'
            # Regex for time item (\d{2}[:]\d{2}[:]\d{2})
            time_re = '(\d{2}[:]\d{2}[:]\d{2})'

            with open('out.txt', 'rt') as file:

                for line in file.readlines():

                    # Write lines that are more than 40 chars long
                    if len(line) <= 40:
                        continue
                    
                    # Find date and time, and total time in seconds
                    date = re.findall(date_re, line)
                    time = re.findall(time_re, line)
                    tot_time = time_into_seconds(time[0], date[0])

                    line = line.replace(date[0] + ',' + time[0], str(tot_time))

                    # Write line to file
                    fout.write(line)

            fout.close()
            os.chdir('../')
            print('Converting date-time to seconds for `{}` folder...'.format(dir))
        
        os.chdir('../')

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