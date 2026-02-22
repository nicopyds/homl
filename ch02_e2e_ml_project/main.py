import os
from pathlib import Path

from zlib import crc32

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

import matplotlib.pyplot as plt

# /Users/nicolaepopescul/code/streams/homl/ch02_e2e_ml_project
FILE_PATH = os.path.abspath(__file__)
datasets_path = os.path.join(os.path.dirname(FILE_PATH), "datasets")


def shuffle_and_split_with_strata(df, test_size):

    shuff_ = StratifiedShuffleSplit(n_splits=10, test_size=test_size, random_state=175)

    splits_ = []

    # tiene un parámetro de groups
    # pero siempre se ignora
    # en nuestro caso, el strata lo tenemos en la columan de income_cat
    # y se "respeta" los % en base a esta columna
    for train_index, test_index in shuff_.split(X=df, y=df["income_cat"]):
        splits_.append((df.iloc[train_index], df.iloc[test_index]))

    return splits_


# la podemos sustituir por una de scikit-learn
def shuffle_and_split_data(df: pd.DataFrame, test_ratio: float, rng):

    shuffled_indeces = rng.permutation(len(df))
    # float, lo casteamos a int para tener un número entero

    test_set_size = int(len(df) * test_ratio)

    test_indeces = shuffled_indeces[:test_set_size]
    train_indeces = shuffled_indeces[test_set_size:]

    # iloc: index location va a utilizar la posición del index de
    # pandas por tanto podemos sin miedo usar train_indeces
    # pero no debería usar el loc (que es la location por nombre en pandas)
    return df.iloc[train_indeces], df.iloc[test_indeces]


def is_id_in_test_set(identifier, test_ratio):
    return crc32(np.int64(identifier)) < test_ratio * 2**32


def shuffle_and_split_with_hash(df: pd.DataFrame, test_ratio: float, id_column: str):
    ids = df[id_column]
    in_test_set = ids.apply(lambda id_: is_id_in_test_set(id_, test_ratio))

    return df[~in_test_set], df[in_test_set]


#    print("boolean mask")
#    print(in_test_set.sum())
#
#    in_test_set = ids.apply(
#        lambda id_: crc32(np.int64(id_)) < (test_ratio * 2 * 32)
#    )
#
#    print("boolean mask")
#    print(in_test_set.sum())


def load_data():
    tarball_path = Path("./datasets/housing.csv")
    return pd.read_csv(tarball_path)


def plot_data(df: pd.DataFrame):
    df["ocean_proximity"].value_counts().plot(kind="bar")
    t = df["income_cat"].value_counts().sort_index()
    print(t)
    t.plot(kind="bar")

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
    df.reset_index(inplace=True)

    df["income_cat"] = pd.cut(
        x=df["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    # instance_1 = df.iloc[0, :]
    # target_1 = instance_1.pop("median_house_value")

    #    print(df.head().T)
    #    print(df.info())
    #    print(df.describe().round(2).T)

    #     plot_data(df=df)

    # esta sería otra forma muy útil de hacer el split
    # usando un hash

    rng = np.random.default_rng(seed=175)

    if os.path.isfile(os.path.join(datasets_path, "X_train.csv")):
        X_train = pd.read_csv(os.path.join(datasets_path, "X_train.csv"))
        X_test = pd.read_csv(os.path.join(datasets_path, "X_test.csv"))

    else:
        # varias formas de hacer el split de nuestros datos
        X_train, X_test = shuffle_and_split_data(df=df, test_ratio=0.2, rng=rng)
        # la primera vez, guardamos los ficheros
        # y luego siempre entramos en el if inicial
        X_train.to_csv(os.path.join(datasets_path, "X_train.csv"))
        X_test.to_csv(os.path.join(datasets_path, "X_test.csv"))

        X_train, X_test = shuffle_and_split_with_hash(
            df=df, test_ratio=0.2, id_column="index"
        )
        X_train.to_csv(os.path.join(datasets_path, "X_train_hash.csv"))
        X_test.to_csv(os.path.join(datasets_path, "X_test_hash.csv"))

        X_train, X_test = train_test_split(df, test_size=0.2, random_state=42)
        X_train.to_csv(os.path.join(datasets_path, "X_train_sk.csv"))
        X_test.to_csv(os.path.join(datasets_path, "X_test_sk.csv"))

        splits = shuffle_and_split_with_strata(df=df, test_size=0.2)
        print(splits)


if __name__ == "__main__":
    main()
