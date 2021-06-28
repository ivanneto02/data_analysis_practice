import os
import re
import pandas

def main():

    data_path = 'D:/Documents/Research/UMBC/geolife_data/Data2'

    os.chdir(data_path)

    for dir in next(os.walk('.'))[1]:
        
        os.chdir('./' + dir)

        if 'labels.txt' in next(os.walk('.'))[2]:
            
            fout = open('labelsOut.txt', 'wt')

            with open('labels.txt', 'rt') as file:

                # Regex for date item (\d{4})[\/](\d{2})[\/](\d{2})
                # Regex for time item (\d{2})[:](\d{2})[:](\d{2})

                for line in file:
                    
                    # list of match instances
                    date_list = re.findall('(\d+[\/]\d+[\/]\d{2})', line)
                    time_list = re.findall('(\d+[:]\d+[:]\d{2})', line)

                    if (len(date_list) > 0) and (len(time_list) > 0):

                        start_date, start_time = date_list[0], time_list[0]
                        end_date, end_time = date_list[1], time_list[1]

                        start_int = time_into_seconds(start_time, start_date)
                        end_int = time_into_seconds(end_time, end_date)

                        line = line.replace(start_date + " " + start_time, str(start_int))
                        line = line.replace(end_date + " " + end_time, str(end_int))
                        line = line.replace('\t', ',')

                        fout.write(line)
                    else:
                        fout.write('"Start Time","End Time","Transportation Mode"\n')

            print(dir)         

        os.chdir('..')

    # Now we have made new `labelsOut.txt` files in which we have our ranges
    # We now have to group each time in our `out.txt` files to these ranges
    
    # Go to our data
    os.chdir(data_path)

    for dir in next(os.walk('.'))[1]:
        
        os.chdir('./' + dir)

        if 'labelsOut.txt' in next(os.walk('.'))[2]:
            
            fout = open('labelsAndLoc.txt', 'wt')

            with open('labelsOut.txt', 'rt') as file:

                # For this step, we have to find locations in out.txt that belong to the time ranges in labelsOut.txt
                pass

            print(dir)         

        os.chdir('..')
    

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