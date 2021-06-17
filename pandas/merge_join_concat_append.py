import pandas as pd

def main():
    df_1 = pd.DataFrame({'id' : [1, 2, 3, 4], 'name': ['hamza', 'farhan', 'vicky', 'jacob'], 'rating' : [3.5, 5.0, 4.8, 7.8]})

    #print(df_1)

    df_2 = pd.DataFrame(columns=['id', 'address', 'score'], data=[[1, 'kwl', 45], [2, 'nankna', 50], [3, 'kwl', 38], [4, 'dfs', 45]])

    #print(df_2)

    df_3 = pd.merge(df_1, df_2)

    #print(df_3)

    df_2 = pd.DataFrame(columns=['id', 'address', 'score'], data=[[1, 'kwl', 45], [2, 'nankna', 50], [3, 'kwl', 38], [3, 'dfs', 45]])

    df_4 = pd.merge(df_1, df_2, left_index=True, right_index=True)

    #print(df_4)

    df_1 = pd.DataFrame({'id' : [1, 2, 3, 4], 'name': ['hamza', 'farhan', 'vicky', 'jacob'], 'rating' : [3.5, 5.0, 4.8, 7.8]})
    df_2 = pd.DataFrame(columns=['id', 'address', 'score'], data=[[1, 'kwl', 45], [2, 'nankna', 50], [3, 'kwl', 38], [3, 'dfs', 45]])

    df_3 = df_2.join(df_1, sort=True, lsuffix='_sort')
    #print(df_3)

    df_3 = pd.concat([df_1, df_2], axis=1)
    #print(df_3)

    df_3 = df_1.append(df_2, ignore_index=False, sort=True)
    #print(df_3)

    df_3 = df_1.join(df_2, how='inner', lsuffix='a')
    #print(df_3)

    df_3 = df_1.join(df_2, how='outer', lsuffix='a')
    #print(df_3)

    df_3 = df_1.join(df_2, how='left', lsuffix='a')
    #print(df_3)

    df_3 = df_1.join(df_2, how='right', lsuffix='a')
    print(df_3)

if __name__ == "__main__":
    main()