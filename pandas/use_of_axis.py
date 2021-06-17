import pandas as pd

def main():
    pd.options.display.max_rows = 100

    housing_df = pd.read_csv('../databases/housing.csv')

    #print(housing_df.head(20))

    print(
        "STD, AXIS = 1:\n",
        housing_df.std(axis=1),
        "\n\nSTD, AXIS = 0:\n",
        housing_df.std(axis=0),
        "\n\nMEAN, AXIS = 1:\n",
        housing_df.mean(axis=1),
        "\n\nMEAN, AXIS = 0:\n",
        housing_df.mean(axis=0),
        "\n\nMAX, AXIS = 1:\n",
        housing_df.max(axis=1),
        "\n\nMAX, AXIS = 0:\n",
        housing_df.max(axis=0),
        "\n\nMIN, AXIS = 1:\n",
        housing_df.min(axis=1),
        "\n\nMIN, AXIS = 0:\n",
        housing_df.min(axis=0),
        "\n\nSUM, AXIS = 1:\n",
        housing_df.sum(axis=1),
        "\n\nSUM, AXIS = 0:\n",
        housing_df.sum(axis=0),
        "\n\nDROP, AXIS = 1:\n",
        housing_df.drop('Net.Operating.Income',axis=1),
        "\n\nDROP, AXIS = 0:\n",
        housing_df.drop(2,axis=0)
        )

if __name__ == "__main__":
    main()