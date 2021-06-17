import pandas as pd
import matplotlib.pyplot as plt

def main():

    pd.options.display.max_rows = 100

    ufo_df = pd.read_csv('./databases/uforeports.csv')

    #print("Printing time...")
    #print(ufo_df.Time.head(5))

    ufo_df['Time'] = pd.to_datetime(ufo_df.Time)

    #print(ufo_df.head(5))
    #print(ufo_df.Time.dt.day_name())

    ufo_df['Year'] = ufo_df.Time.dt.year

    plt.plot(ufo_df.Year.value_counts().sort_index())
    plt.show()

if __name__ == "__main__":
    main()