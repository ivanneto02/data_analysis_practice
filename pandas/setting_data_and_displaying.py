import pandas as pd

def main():

    # Set options
    pd.set_option('display.max_rows', 50)
    pd.set_option('display.max_colwidth', 100)
    pd.set_option('display.precision', 6)

    titanic_df = pd.read_csv('../databases/titanic.csv')
    print(titanic_df.head(10))

    titanic_df['X'] = titanic_df.Fare * 10000
    titanic_df['Y'] = titanic_df.Age * 10000

    print(titanic_df)

if __name__ == "__main__":
    main()