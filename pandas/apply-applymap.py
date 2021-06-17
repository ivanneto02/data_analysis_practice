import pandas as pd

def main():
    titanic_df = pd.read_csv('../databases/titanic.csv')

    print(titanic_df)

    titanic_df['Name Length'] = titanic_df.Name.apply(len)

    print(titanic_df.loc[0:4, ['Name']])

    nm = ''

    # Last names
    nms = []

    sms = ''

    # Separates last names and first names
    for name in titanic_df.Name:
        nm = name.split(',')
        nms.append(nm[0])
    
    # Makes last name column
    titanic_df['Last Name'] = [x for x in nms]

    #print(titanic_df.loc[0:, ['Name', 'Last Name']])

    #print(titanic_df.Name.tail())

    titanic_df['Age'].apply(float)

    titanic_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Last Name'], axis=1, inplace=True)
    titanic_df.fillna(titanic_df.Age.mean(), axis=1, inplace=True)

    print(titanic_df)

    # Normalize

    titanic_df_mean = titanic_df.mean()
    titanic_df_std = titanic_df.std()
    titanic_df_normalized = (titanic_df - titanic_df_mean) / titanic_df_std

    print(titanic_df_normalized)

if __name__ == "__main__":
    main()