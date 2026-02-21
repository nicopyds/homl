from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data():
    tarball_path = Path("./datasets/housing.csv")
    return pd.read_csv(tarball_path)


def plot_data(df: pd.DataFrame):
    df["ocean_proximity"].value_counts().plot(kind="bar")

    nr_rows = df.shape[0]
    # para decidir el número de bins
    # https://en.wikipedia.org/wiki/Sturges%27s_rule
    nr_bins = int((1 + np.log2(nr_rows)))
    print(nr_bins)
    df.hist(bins=nr_bins)
    df.hist(bins=50)

    nums = df.select_dtypes(include=np.number)
    nums = nums.dropna()
    nums_log = nums.applymap(np.log1p)
    nums_log.hist(bins=50)

    plt.show()


def main():
    df = load_data()

    instance_1 = df.iloc[0, :]
    # target_1 = instance_1.pop("median_house_value")

    print(df.head().T)
    print(df.info())
    print(df.describe().round(2).T)

    plot_data(df=df)


if __name__ == "__main__":
    main()
