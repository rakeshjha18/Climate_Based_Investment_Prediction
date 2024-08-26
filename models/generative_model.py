import tensorflow as tf
from tensorflow.python.keras import layers, models


def build_vae(input_shape):
    # Encoder
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(inputs)
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(16, activation="relu")(x)

    z_mean = layers.Dense(2, name="z_mean")(x)
    z_log_var = layers.Dense(2, name="z_log_var")(x)

    # Latent space
    z = layers.Lambda(sampling, output_shape=(2,), name="z")([z_mean, z_log_var])

    # Decoder
    decoder_inputs = layers.Input(shape=(2,))
    x = layers.Dense(7 * 7 * 64, activation="relu")(decoder_inputs)
    x = layers.Reshape((7, 7, 64))(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)

    # Models
    encoder = models.Model(inputs, [z_mean, z_log_var, z], name="encoder")
    decoder = models.Model(decoder_inputs, outputs, name="decoder")
    vae = models.Model(inputs, decoder(encoder(inputs)[2]), name="vae")

    return vae


def sampling(args):
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon


if __name__ == "__main__":
    vae = build_vae((28, 28, 1))
    vae.compile(optimizer="adam", loss="binary_crossentropy")
    # You will add training code here.
