import pandas as pd

def main():
    chip_df = pd.read_table('./databases/chipotle.csv')

    print(chip_df)
    
    print(chip_df.iloc[0:2, 0:2])

if __name__ == "__main__":
    main()