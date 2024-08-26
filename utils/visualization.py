import matplotlib.pyplot as plt


def plot_history(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.show()


if __name__ == "__main__":
    # Example usage
    # history = ... (get this from model training)
    # plot_history(history)
    pass
