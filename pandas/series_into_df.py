import numpy as np
import pandas as pd

def main():

    titanic_df = pd.read_csv('../databases/titanic.csv')

    new_series = pd.Series(list(np.linspace(1, 4622, 4622) * np.random.randint(255)), index=list(np.linspace(1, 4622, 4622)), name='Khan')

    titanic_df = pd.concat([titanic_df, new_series], axis=1)

    print(titanic_df.Khan)
    print(titanic_df)

if __name__ == "__main__":
    main()