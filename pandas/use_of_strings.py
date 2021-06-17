import pandas as pd

def main():

    chipotle_df = pd.read_csv('./databases/chipotle.csv')

    #print(chipotle_df.item_name.str.upper())
    #print(chipotle_df.item_name.str.lower())

    chipotle_df['item_name'] = chipotle_df['item_name'].apply(lambda x: x.lower())

    condition1 = chipotle_df.item_name.str.contains('burrito')

    #print(chipotle_df[condition1].head(10))

    chipotle_df['choice_description'].replace('[', '')

    chipotle_df['choice_description'].replace('[\[\]]', '')

    chipotle_df['choice_description'].replace('[', '').str.replace(']', '')

    print(chipotle_df[condition1].head(10))

if __name__ == "__main__":
    main()