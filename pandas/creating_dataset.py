import pandas as pd
import numpy as np

def main():

    # Make first dataframe
    df_1 = pd.DataFrame([[1, 2, 3], [6, 7, 9]], columns=['a', 'b', 'c'])
    print(df_1)

    # Make second dataframe
    df_2 = pd.DataFrame({'id':np.arange(20, 41, 1), 'age':np.linspace(80, 100, 21)})
    #print(df_2)

    # Test how linspace works
    test_linspace = np.linspace(50, 100, 10)
    #print(test_linspace)
    print('linspace size:', test_linspace.size)

    # Test how arange works
    test_arange = np.arange(20, 50, 1)
    #print(test_arange)
    print('arange size:', test_arange.size)

if __name__ == "__main__":
    main()