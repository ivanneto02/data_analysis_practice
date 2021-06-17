import pandas as pd


def main():
    df_1 = pd.read_csv('../databases/titanic.csv')
    df_2 = pd.read_table('../databases/titanic.csv')
    df_3 = pd.read_fwf('../databases/titanic.csv', sep=',')

    print(df_1)
    print(df_2)
    print(df_3)

if __name__ == "__main__":
    main()