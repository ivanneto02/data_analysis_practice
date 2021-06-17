import pandas as pd
import numpy as np

def main():
    housing_df = pd.read_csv('../databases/housing.csv')

    print(housing_df.select_dtypes([np.object]))

if __name__ == "__main__":
    main()