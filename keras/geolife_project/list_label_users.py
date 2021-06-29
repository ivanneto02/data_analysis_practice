import os

def main():

    data_path = 'D:/Documents/Research/UMBC/geolife_data/Data2'

    os.chdir(data_path)

    for dir in next(os.walk('.'))[1]:

        os.chdir(dir)
            
        if 'labels.txt' not in next(os.walk('.'))[2]:
            print(dir, '- NOT in here')
            os.chdir('../')
            continue
        
        print(dir, '- IN HERE')
        os.chdir('../')

if __name__ == '__main__':
    main()
