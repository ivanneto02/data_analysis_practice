import pandas as pd

def main():
    ufo_df = pd.read_csv('./databases/uforeports.csv')
    print(ufo_df.shape)
    print(ufo_df.describe())

    ufo_df_dropped = ufo_df.dropna()

    print("\'City\' null values...")
    print(ufo_df_dropped['City'].isnull())

    print('\'Colors Reported\' null values...')
    print(ufo_df_dropped['Colors Reported'].isnull())

    print('\'Shape Reported\' null values...')
    print(ufo_df_dropped['Shape Reported'].isnull())

    print('\'State\' null values...')
    print(ufo_df_dropped['State'].isnull())

    print('\'Time\' null values...')
    print(ufo_df_dropped['Time'].isnull())

    ufo_df_undropped = ufo_df

    # Inplace
    print(ufo_df['Colors Reported'].value_counts(dropna=True))
    print(ufo_df['Colors Reported'].value_counts(dropna=False))

    print('Dropped...')
    print(ufo_df_dropped.shape)
    print('Undropped...')
    print(ufo_df_undropped.shape)


if __name__ == "__main__":
    main()