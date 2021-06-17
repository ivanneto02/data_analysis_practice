import pandas as pd

def main():

    housing_df = pd.read_csv('./databases/housing.csv')

    housing_df.dropna(inplace=True)
    housing_df.rename(columns={'Neighborhood':'place'}, inplace=True)

    to_print = housing_df.head(10)

    print(to_print)

if __name__ == "__main__":
    main()