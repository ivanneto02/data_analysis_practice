import pandas as pd

def main():
    housing_df = pd.read_csv('./databases/housing.csv')

    print(housing_df.iloc[2:, 1:4])

    print(housing_df.iloc[2:3, 2:3])

    print(housing_df.iloc[lambda x: x.index % 2 != 0])

    print(housing_df.iloc[[0, 2], [1, 3]])
    
    print(housing_df.iloc[:, lambda housing_df: [0, 2]])

    print(housing_df.iloc[[0]])

    print(housing_df.iloc[[0, 1]])

if __name__ == "__main__":
    main()