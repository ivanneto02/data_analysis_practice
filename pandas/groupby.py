import pandas as pd
import matplotlib.pyplot as plt

def main():
    titanic_df = pd.read_csv('./databases/titanic.csv')

    print(titanic_df)

    # Getting the mean of every value, grouped by gender
    titanic_s_group = titanic_df.groupby('Sex').mean()
    print(titanic_s_group)

if __name__ == "__main__":
    main()