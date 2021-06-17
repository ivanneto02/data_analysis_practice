import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    pew = pd.read_csv('../databases/pew.csv')
    drinks = pd.read_csv('../databases/drinks.csv')

    #print(pew.head(15))
    #print(drinks.head(10))

    #print(pd.melt(pew, id_vars='religion', var_name='OBJECT', value_name='INTEGER').head(15))

    billboard_df = pd.read_csv('../databases/billboard.csv')

    #print(billboard_df.head(10))

    billboard_df = pd.melt(billboard_df, id_vars=['year', 'artist.inverted', 'track', 'date.entered'])

    #print(billboard_df.head(10))

    ebola = pd.read_csv('../databases/country_timeseries.csv')

    #print(ebola.head(10))

    pd.options.display.max_rows = 50

    ebola = pd.melt(ebola, id_vars=['Date', 'Day'], value_name='deaths', )

    #print(ebola.tail(50))

    ebola[['First', 'Second']]=ebola.variable.str.split('_', expand=True)

    #ebola.to_csv('./output.csv')

    #plt.show(ebola[ebola.First == 'Cases'].groupby(['First', 'Second']).deaths.sum().plot(kind='bar'))
    #plt.show(ebola[ebola.First == 'Deaths'].groupby(['First', 'Second']).deaths.sum().plot(kind='bar'))
    #plt.show(ebola[ebola.First == 'Cases'].groupby(['First']).deaths.sum().plot(kind='bar'))

    ebola_pt = ebola.pivot_table(index=['Date', 'Second'], columns=['First'], values='deaths', dropna=True)
    #print(ebola_pt.head(10))

    #ebola_pt.to_csv('./output.csv')

    ebola['id'] = range(ebola.shape[0])

    print(ebola.head(10))

    power = ebola.drop(['Day', 'variable'], axis=1)

    print(power.head(10))

    #print(power.head(30))

    new_df = ebola.loc[:, ['Day', 'variable']]

    print(new_df.head(10))
    print('new_df shape:', new_df.shape)

    new_power_df = power.merge(new_df, left_on='deaths', right_on='Day')

    print(new_power_df.head(10))

if __name__ == "__main__":
    main()