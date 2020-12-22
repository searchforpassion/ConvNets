from tensorflow import keras
from tensorflow.keras import layers

def make_model(input_shape=(224,224,3), num_classes=1000):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(64, kernel_size=(3, 3), strides=1, padding="same", activation="relu")(inputs)
    x = layers.Conv2D(64, kernel_size=(3, 3), strides=1, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid")(x)
    
    x = layers.Conv2D(128, kernel_size=(3, 3), strides=1, padding="same", activation="relu")(x)
    x = layers.Conv2D(128, kernel_size=(3, 3), strides=1, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid")(x)

    x = layers.Conv2D(256, kernel_size=(3, 3), strides=1, padding="same", activation="relu")(x)
    x = layers.Conv2D(256, kernel_size=(3, 3), strides=1, padding="same", activation="relu")(x)
    x = layers.Conv2D(256, kernel_size=(3, 3), strides=1, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid")(x)

    x = layers.Conv2D(512, kernel_size=(3, 3), strides=1, padding="same", activation="relu")(x)
    x = layers.Conv2D(512, kernel_size=(3, 3), strides=1, padding="same", activation="relu")(x)
    x = layers.Conv2D(512, kernel_size=(3, 3), strides=1, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid")(x)

    x = layers.Conv2D(512, kernel_size=(3, 3), strides=1, padding="same", activation="relu")(x)
    x = layers.Conv2D(512, kernel_size=(3, 3), strides=1, padding="same", activation="relu")(x)
    x = layers.Conv2D(512, kernel_size=(3, 3), strides=1, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid")(x)

    x = layers.Flatten()(x)
    x = layers.Dense(4096, activation="relu")(x)
    x = layers.Dense(4096, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    return keras.Model(inputs, outputs)
    
if __name__ == '__main__':
    model = make_model(input_shape=(224,224,3), num_classes=1000)
    model.summary()
    keras.utils.plot_model(model, show_shapes=True)