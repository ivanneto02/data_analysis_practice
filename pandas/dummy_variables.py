import pandas as pd

def main():

    pd.options.display.max_rows = 50

    titanic_df = pd.read_csv('../databases/titanic.csv')

    #print(titanic_df.head(10))

    # Encode sex
    titanic_df['sex_encoded'] = titanic_df.Sex.map({'male' : 1, 'female' : 0})

    #print(titanic_df.head(10))

    titanic_df['SEX'] = pd.get_dummies(titanic_df.Sex).iloc[:,[1]]

    #print(titanic_df.loc[:,['SEX', 'sex_encoded']])

    titanic_df_embarked_encoding = pd.get_dummies(titanic_df.Embarked, prefix='embarked')

    titanic_df_concat = pd.concat([titanic_df, titanic_df_embarked_encoding], axis=1)

    #print(titanic_df_concat.head(10))

    print(titanic_df_concat.loc[:, ['embarked_Q', 'embarked_S', 'embarked_C']])

    sex_embark_df = pd.get_dummies(titanic_df, columns=['Sex', 'Embarked'], drop_first=False)
    
    print(sex_embark_df.head(10))

if __name__ == "__main__":
    main()