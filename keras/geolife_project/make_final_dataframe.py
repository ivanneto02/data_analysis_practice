import os
import pandas as pd

def main():
    
    pd.options.display.max_rows = 500

    data_path = 'D:/Documents/Research/UMBC/geolife_data/Data2'

    os.chdir(data_path + '/final_dataframes')

    base_df = pd.DataFrame()

    for csv_file in next(os.walk('.'))[2]:

        curr_df = pd.read_csv('./' + csv_file)

        base_df = base_df.append(curr_df)

    base_df.rename({'Unnamed: 0' : 'Row'}, inplace=True, axis=1)

    print(base_df)

    base_df.to_csv('./per_user/final_df.csv')

if __name__ == '__main__':
    main()