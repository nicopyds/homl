from pathlib import Path
from numpy import median
import pandas as pd


def load_data():
    tarball_path = Path("./datasets/housing.csv")
    return pd.read_csv(tarball_path)


def main():
    load_data()

if __name__ == "__main__":
    main()
