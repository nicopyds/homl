from pathlib import Path
import pandas as pd


def load_data():
    tarball_path = Path("./datasets/housing.csv")
    return pd.read_csv(tarball_path)


def main():
    df = load_data()
    instance_1 = df.iloc[0, :]
    target_1 = instance_1.pop("median_house_value")
    print(instance_1)
    print(target_1)


if __name__ == "__main__":
    main()
