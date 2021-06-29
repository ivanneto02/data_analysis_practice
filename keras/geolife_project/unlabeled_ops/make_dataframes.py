import os
import pandas as pd

def main():

    #make_bases()
    #make_individual()

    make_final_dataframes()

def make_final_dataframes():

    base_df = 'D:/Documents/Research/UMBC/geolife_data/Data2/final_dataframes'
    labeled_dp = 'D:/Documents/Research/UMBC/geolife_data/Data2/final_dataframes/labeled/per_user'
    unlabeled_dp = 'D:/Documents/Research/UMBC/geolife_data/Data2/final_dataframes/unlabeled/per_user'

    # For labeled dataframes...
    print('Making final labeled dataframe...')

    os.chdir(labeled_dp)
    
    final_labeled_df = pd.DataFrame()

    for file in next(os.walk('.'))[2]:
        curr_df = pd.read_csv(file)
        final_labeled_df = final_labeled_df.append(curr_df)

    final_labeled_df.drop('Unnamed: 0', inplace=True, axis=1)
    final_labeled_df.to_csv('../final_labeled_df.csv')

    # For unlabeled dataframes...
    print('Making final unlabeled dataframe...')

    os.chdir(unlabeled_dp)

    final_unlabeled_df = pd.DataFrame()

    for file in next(os.walk('.'))[2]:
        curr_df = pd.read_csv(file)
        final_unlabeled_df = final_unlabeled_df.append(curr_df)

    final_unlabeled_df.drop('Unnamed: 0', inplace=True, axis=1)
    final_unlabeled_df.to_csv('../final_unlabeled_df.csv')

    print('Merging both...')

    os.chdir(base_df)

    first = pd.read_csv('./labeled/final_labeled_df.csv')
    second = pd.read_csv('./unlabeled/final_unlabeled_df.csv')

    final = first.append(second).drop('Unnamed: 0', axis=1)

    final.to_csv('./final_df.csv')

    print('Done!')

def make_individual():
    data_path = 'D:/Documents/Research/UMBC/geolife_data/Data2'

    labeled_dp = 'D:/Documents/Research/UMBC/geolife_data/Data2/final_dataframes/labeled/per_user'
    unlabeled_dp = 'D:/Documents/Research/UMBC/geolife_data/Data2/final_dataframes/unlabeled/per_user'

    curr_traj = 10000000000

    os.chdir(data_path)

    for dir in next(os.walk('.'))[1]:

        if (dir != 'final_dataframes'):

            print('Creating dataframe for user {}...'.format(dir))

            os.chdir(dir)

            curr_df = pd.read_csv('final_df_{}.csv'.format(dir))

            curr_df['user_id'] = dir
            curr_df['trajectory_id'] = [i + curr_traj for i in range(0, len(curr_df))]
            curr_traj += len(curr_df)

            curr_df = curr_df[['user_id', 'trajectory_id', 'Start Time', 'End Time', 'Start_lat', 'End_lat', 'Start_long', 'End_long', 'label']]

            if 'labels.txt' in next(os.walk('.'))[2]:

                curr_df.to_csv(labeled_dp + '/final_df_{}.csv'.format(dir))

            else:

                curr_df.to_csv(unlabeled_dp + '/final_df_{}.csv'.format(dir))


        os.chdir('../')

def make_bases():

    data_path = 'D:/Documents/Research/UMBC/geolife_data/Data2'

    os.chdir(data_path)

    for dir in next(os.walk('.'))[1]:

        os.chdir(dir)

        if ('labels.txt' not in next(os.walk('.'))[2]) and (dir != 'final_dataframes'):

            curr_df = pd.read_csv('./trajectories.txt')

            new_df = pd.DataFrame()

            new_df['Start Time'] = curr_df['Start Time']
            new_df['End Time'] = curr_df['End Time']
            new_df['Start_lat'] = curr_df['Start_lat']
            new_df['End_lat'] = curr_df['End_lat']
            new_df['Start_long'] = curr_df['Start_long']
            new_df['End_long'] = curr_df['End_long']
            new_df['label'] = curr_df['label']

            new_df.to_csv('final_df_{}.csv'.format(dir))
            print('Making dataframe for user {}...'.format(dir))

        os.chdir('../')


if __name__ == '__main__':
    main()