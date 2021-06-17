import pandas as pd
import matplotlib.pyplot as plt

def main():

    pd.set_option('display.max_colwidth', 100)

    chipotle_df = pd.read_csv('../databases/chipotle.csv')

    print(chipotle_df.head(5))

    chipotle_df.index.name = None

    print(chipotle_df.head(5))

    chipotle_df.index.name = 'nice'

    print(chipotle_df.head(5))

    chipotle_df.set_index('item_name', inplace=True)

    print(chipotle_df.head(5))

    chipotle_df.reset_index(inplace=True)

    print(chipotle_df.head(5))

if __name__ == "__main__":
    main()