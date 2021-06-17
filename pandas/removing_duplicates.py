import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    my_df = pd.DataFrame({ 'a': [1, 0, 9, 1, 1, 9, 1, 2, 3, 2],
                           'b': [5, 8, 6, 5, 5, 6, 5, 2, 2, 2]})

    print("Without removing duplicates...")
    print(my_df)

    print("\nRemoving duplicates...")
    print(my_df.drop_duplicates())

if __name__ == "__main__":
    main()