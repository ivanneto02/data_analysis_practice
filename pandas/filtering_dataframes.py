import pandas as pd

def main():
    clg_df = pd.read_csv('./databases/banklist.csv')

    pd.options.display.max_rows = 100

    print(clg_df[(clg_df.CERT > 1000) & (clg_df.ST == 'OH')].head(100))
    print(clg_df.head(20))

if __name__ == "__main__":
    main()