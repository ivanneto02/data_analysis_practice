import pandas as pd
import numpy as np

def main():
    wine_df = pd.read_csv('../databases/wine.csv')

    print(wine_df)

    print(wine_df.memory_usage(deep=True))

    housing_df = pd.read_csv('../databases/housing.csv')
    
    print(housing_df)

    print(housing_df.memory_usage(deep=True))

if __name__ == "__main__":
    main()