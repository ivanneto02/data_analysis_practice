import os

def main():
    '''
    The aim of this file is to convert all of the .PLT files into a single out.txt file,
    but only for users that do not have labels inside of their data folders.
    '''

    data_path = 'D:/Documents/Research/UMBC/geolife_data/Data2'

    os.chdir(data_path)

    for dir in next(os.walk('.'))[1]:

        os.chdir('./' + dir)

        if ('labels.txt' not in next(os.walk('.'))[2]) and (dir != 'final_dataframes'):

            # Access `Trajectory` folder
            os.chdir('./Trajectory/')

            fout = open('out.txt', 'wt')
            fout.write('lat, long, ?1, ?2, alt, date, time\n')

            # For each .PLT file
            for file in next(os.walk('.'))[2]:
                
                if file[-4:] == '.plt':
                    
                    with open(file, 'r') as pltfile:

                        for line in pltfile.readlines():
                            
                            if len(line) < 43:
                                continue
                        
                            # Write lines longer than 43 chars
                            fout.write(line)

            fout.close()
            os.chdir('../')
            print('Converging files for `{}` folder...'.format(dir))
        
        os.chdir('../')

if __name__ == '__main__':
    main()