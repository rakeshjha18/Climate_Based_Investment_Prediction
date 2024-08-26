import pandas as pd


def load_data(path):
    return pd.read_csv(path)


if __name__ == "__main__":
    data = load_data("../data/processed/processed_data.csv")
    print(data.head())
