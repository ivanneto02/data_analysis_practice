import pandas as pd

def main():

    housing_df = pd.read_csv('./databases/satgpa.csv')

    housing_df_copy = housing_df.copy()

    to_print = housing_df_copy[housing_df_copy['fy_gpa'] >= 3.80].head(10)

    print(to_print)

if __name__ == "__main__":
    main()