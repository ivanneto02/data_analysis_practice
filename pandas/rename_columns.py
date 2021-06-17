import pandas as pd

def main():

    pd.options.display.max_rows = 100

    my_df = pd.read_csv('../databases/housing.csv')

    naming_list = ['NEIGHBORHOOD', 'BUILDING_CLASSIFICATION', 'TOTAL_UNITS', 'YEAR_BUILT', 'GROSS_SQFT', 'ESTIMATED_GROSS_INCOME', 'GROSS_INCOME_PER_SQFT', 'ESTIMATED_EXPENSE',
                   'EXPENSE_PER_SQFT', 'NET_OPERATING_INCOME', 'FULL_MARKET_VALUE', 'MARKER_VALUE_PER_SQFT', 'BORO']

    print('Original dataframe...')
    print(my_df.head(10))

    my_df = pd.read_csv('../databases/housing.csv', names=naming_list, header=0)
    print('\nDataframe after column name change...')
    print(my_df)
    

    new_df = pd.read_csv('../databases/housing.csv')
    print('\nNew dataframe without changing columns...')
    print(new_df)

    # Change columns
    new_df.columns = naming_list

    print('\nHaving changed the columns...')
    print(new_df)

    new_df_2 = pd.read_csv('../databases/housing.csv')
    new_df_2 = new_df_2.rename(columns={'Neighborhood': 'NEIGHBORHOODS'})

    print(new_df_2)

if __name__ == "__main__":
    main()