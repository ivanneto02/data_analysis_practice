import pandas as pd

def main():

    pd.options.display.max_rows = 100

    gpa_df = pd.read_csv('../databases/satgpa.csv')

    to_print_df = gpa_df[(gpa_df['fy_gpa'] >= 4.0) & (gpa_df['hs_gpa'] >= 4.0)]

    print('To select characteristics such as gpa > 3.0...')
    print(to_print_df.head(20))

    print('\nTo select max of any numeric column...')

    condition1 = gpa_df['fy_gpa'] == max(gpa_df['fy_gpa'])
    condition2 = gpa_df['sat_sum'] == max(gpa_df['sat_sum'])

    to_print_df = gpa_df[condition1 & condition2]
    print(to_print_df)

if __name__ == "__main__":
    main()