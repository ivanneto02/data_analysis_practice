import pandas as pd
import matplotlib.pyplot as plt

def main():

    ufo_df = pd.read_csv('./databases/uforeports.csv')

    title = pd.read_csv('./databases/uforeports.csv', nrows = 0)

    # DF that holds values that are duplicates
    dd = ufo_df.duplicated(title)

    # Indexes to drop (?)
    drop = []

    # Append the indexes of duplicate values into drop
    for i in range(len(dd)):
        if dd[i] == True:
            drop.append(i)

    # Before and after drop() is applied to ufo_df
    print(ufo_df.shape)
    print(ufo_df.drop(drop, axis = 0).shape)

if __name__ == "__main__":
    main()