import pandas as pd
import matplotlib.pyplot as plt

def main():

    chipotle_df = pd.read_csv('../databases/chipotle.csv')

    '''
    print(chipotle_df.head(100))
    print(chipotle_df.describe())
    print(chipotle_df.item_name.describe())
    '''

    # Normalize
    chipotle_df_mean = chipotle_df.mean()
    chipotle_df_std = chipotle_df.std()
    chipotle_df_norm = (chipotle_df - chipotle_df_mean) / chipotle_df_std

    #print(chipotle_df_norm.describe())

    #print(chipotle_df.item_name.value_counts())
    
    dd = chipotle_df.item_name.value_counts(normalize=True)
    print(dd)

    plot_var = chipotle_df.order_id.plot(kind='line')

    plt.show(plot_var)

if __name__ == "__main__":
    main()