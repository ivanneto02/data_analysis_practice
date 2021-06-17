import pandas as pd
import matplotlib.pyplot as plt

def main():
    drinks_df = pd.read_csv('./databases/drinks.csv')

    print(drinks_df['country'][0:])

    plt_obj = drinks_df.beer_servings.plot(kind='hist', rwidth=0.9)

    plt.show(plt_obj)

    drinks_df_copy = drinks_df.copy()

    drinks_df_copy = drinks_df_copy.groupby('continent').sum().drop('total_litres_of_pure_alcohol', axis=1)

    print(drinks_df_copy)

    for df in drinks_df:
        print(drinks_df[df].memory_usage())

if __name__ == "__main__":
    main()