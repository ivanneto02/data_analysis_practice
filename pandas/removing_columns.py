import pandas as pd

def main():

    pd.options.display.max_rows = 100

    titanic_df = pd.read_csv('../databases/titanic.csv')

    print(titanic_df)

    columns_to_drop = ['Name', 'Sex', 'Cabin', 'Embarked', 'Ticket']

    titanic_df_numeric = titanic_df.drop(columns_to_drop, axis=1).dropna()

    print(titanic_df_numeric)

    housing_df = pd.read_csv('../databases/housing.csv')

    print(housing_df)

    housing_numeric_columns = ['Total.Units', 'Year.Built', 'Gross.SqFt', 'Estimated.Gross.Income', 'Estimated.Expense', 'Gross.Income.per.SqFt', 'Expense.per.SqFt', 'Net.Operating.Income', 'Market.Value.per.SqFt']

    housing_numeric_df = housing_df.loc[(housing_df['Year.Built'] >= 2000), housing_numeric_columns]

    print(housing_numeric_df.head(100))

if __name__ == "__main__":
    main()