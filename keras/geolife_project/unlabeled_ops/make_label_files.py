import os
import re

TIME_TRESH = 1800 # 30 minutes in seconds

def main():

    data_path = 'D:/Documents/Research/UMBC/geolife_data/Data2'

    os.chdir(data_path)

    for dir in next(os.walk('.'))[1]:

        os.chdir(dir)

        if ('labels.txt' not in next(os.walk('.'))[2]) and (dir != 'final_dataframes'):

            os.chdir('./Trajectory/')

            fout = open('../trajectories.txt', 'wt')
            fout.write('Start Time,Start_lat,Start_long,End Time,End_lat,End_long,label\n')
            
            fin = open('out_int_dates.txt', 'rt')

            fin_norm = open('out.txt', 'rt')

            write_labels(fin, fin_norm, fout)

            fin_norm.close()
            fin.close()

            fout.close()
            os.chdir('../')

            print('Making `trajectories.txt` for {} folder...'.format(dir))

        os.chdir('../')

'''
Writes to `trajectories.txt` file
'''
def write_labels(fin, fin_norm, fout):
    # fin - int version
    # fin_norm - normal version
    # fout - trajectories.txt

    # Regex for finding floats `(\d+[.]\d+)`

    fin_list = fin.readlines()
    fin_norm_list = fin_norm.readlines()

    fin_list = fin_list[1:]
    fin_norm_list = fin_norm_list[1:]

    start_time = fin_norm_list[0][-20:].strip().replace(',', ' ')
    start_lat, start_long = re.findall('(\d+[.]\d+)', fin_norm_list[0])[0], re.findall('(\d+[.]\d+)', fin_norm_list[0])[1]

    fout.write(start_time + ',' + start_lat + ',' + start_long + ',')

    for i in range(0, len(fin_list) - 1):

        time1_int = int(fin_list[i][-12:].strip())
        time1_str = fin_norm_list[i][-20:].strip().replace(',', ' ')

        time2_int = int(fin_list[i+1][-12:].strip())
        time2_str = fin_norm_list[i+1][-20:].strip().replace(',', ' ')

        if (time2_int - time1_int > TIME_TRESH):

            lat1, long1 = re.findall('(\d+[.]\d+)', fin_norm_list[i])[0], re.findall('(\d+[.]\d+)', fin_norm_list[i])[1]
            lat2, long2 = re.findall('(\d+[.]\d+)', fin_norm_list[i+1])[0], re.findall('(\d+[.]\d+)', fin_norm_list[i+1])[1]

            fout.write(time1_str + ',' + lat1 + ',' + long1 + ',unlabeled\n')
            fout.write(time2_str + ',' + lat2 + ',' + long2 + ',')

    end_time = fin_norm_list[len(fin_norm_list) - 1][-20:].strip().replace(',', ' ')
    end_lat, end_long = re.findall('(\d+[.]\d+)', fin_norm_list[len(fin_list) - 1])[0], re.findall('(\d+[.]\d+)', fin_norm_list[len(fin_list) - 1])[1]

    fout.write(end_time + ',' + end_lat + ',' + end_long + ',unlabeled\n')

if __name__ == '__main__':
    main()