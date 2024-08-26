from sklearn.metrics import mean_squared_error


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Model Mean Squared Error: {mse}")


if __name__ == "__main__":
    from models.predictive_model import build_lstm
    import pandas as pd

    model = build_lstm((10, 1))
    data = pd.read_csv("../data/processed/processed_data.csv")
    X_test, y_test = data.iloc[:, :-1], data.iloc[:, -1]
    evaluate_model(model, X_test, y_test)
