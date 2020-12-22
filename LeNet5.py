from tensorflow import keras
from tensorflow.keras import layers

def make_model(input_shape=(32,32,1), num_classes=10):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(6, 5, strides=1, padding="valid", activation="relu")(inputs)
    x = layers.MaxPooling2D(pool_size=2, strides=2, padding="valid")(x)
    
    x = layers.Conv2D(16, 5, strides=1, padding="valid", activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2, strides=2, padding="valid")(x)

    x = layers.Flatten()(x)
    x = layers.Dense(120, activation="relu")(x)
    x = layers.Dense(84, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    return keras.Model(inputs, outputs)
    
if __name__ == '__main__':
    model = make_model(input_shape=(32,32,1), num_classes=10)
    model.summary()
    keras.utils.plot_model(model, show_shapes=True)
