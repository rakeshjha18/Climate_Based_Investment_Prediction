from tensorflow.python.keras import layers, models


# from tensorflow.python.keras.layers import LSTM


def build_lstm(input_shape):
    model = models.Sequential()
    model.add(layers.LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(layers.LSTM(32, return_sequences=False))
    model.add(layers.Dense(1, activation="linear"))
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


if __name__ == "__main__":
    # Example input shape
    input_shape = (10, 1)  # 10 time steps, 1 feature
    model = build_lstm(input_shape)
    # You will add training code here.
