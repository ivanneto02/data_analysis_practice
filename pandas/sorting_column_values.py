import pandas as pd

def main():

    pd.options.display.max_rows = 50

    housing_df = pd.read_csv('../databases/housing.csv')

    print(housing_df.head(10))
    
    # Sort values of any column

    housing_df = housing_df.sort_values(['Total.Units'], na_position='last').dropna()

    print(housing_df)

if __name__ == "__main__":
    main()