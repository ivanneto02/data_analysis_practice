import pandas as pd

def main():

    pd.options.display.max_rows = 50

    housing_df = pd.read_csv('../databases/housing.csv')

    print(housing_df.head(10))

    print(housing_df[housing_df['Year.Built'] >= 2000].head(10))
    print(housing_df[housing_df.Boro.isin(['Brooklyn', 'Manhattan'])].head(10))

if __name__ == "__main__":
    main()