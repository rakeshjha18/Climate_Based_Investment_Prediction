from models.predictive_model import build_lstm
import pandas as pd


def train_models():
    # Load processed data
    data = pd.read_csv("../data/processed/processed_data.csv")

    # Example training procedure for LSTM
    lstm = build_lstm((10, 1))
    X_train, y_train = data.iloc[:, :-1], data.iloc[:, -1]
    lstm.fit(X_train, y_train, epochs=10, batch_size=32)

    # Save model
    lstm.save("../models/lstm_model.h5")
    print("LSTM Model trained and saved.")


if __name__ == "__main__":
    train_models()
