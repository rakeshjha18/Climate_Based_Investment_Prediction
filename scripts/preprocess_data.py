import pandas as pd
from sklearn.preprocessing import StandardScaler


def preprocess_data(input_path, output_path):
    df = pd.read_csv(input_path)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    df.to_csv(output_path, index=False)
    print("Data preprocessing complete.")


if __name__ == "__main__":
    preprocess_data("../data/raw/environmental_data.csv", "../data/processed/processed_data.csv")
