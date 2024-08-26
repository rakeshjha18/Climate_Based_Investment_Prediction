from tensorflow.python.keras.models import load_model


def save_model(model, path):
    model.save(path)


def load_existing_model(path):
    return load_model(path)


if __name__ == "__main__":
    # Example usage
    model = load_existing_model("../models/lstm_model.h5")
    print("Model loaded successfully.")
