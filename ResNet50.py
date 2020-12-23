from tensorflow import keras
from tensorflow.keras import layers

def identity_bottleneck(input_tensor, kernel_size, filters):
    filters1, filters2, filters3 = filters

    x = layers.Conv2D(filters1, (1, 1),kernel_initializer='he_normal')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1), kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x

def conv_bottleneck(input_tensor, kernel_size, filters, strides=(2, 2)):
    filters1, filters2, filters3 = filters
    
    x = layers.Conv2D(filters1, (1, 1), strides=strides, kernel_initializer='he_normal')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1), kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)

    shortcut = layers.Conv2D(filters3, (1, 1), strides=strides, kernel_initializer='he_normal')(input_tensor)
    shortcut = layers.BatchNormalization()(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x


def make_model(input_shape=(224,224,3), num_classes=1000):
    inputs = keras.Input(shape=input_shape)
    x = layers.ZeroPadding2D(padding=(3, 3))(inputs)
    x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='valid', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.ZeroPadding2D(padding=(1, 1))(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_bottleneck(x, 3, [64, 64, 256], strides=(1, 1))
    x = identity_bottleneck(x, 3, [64, 64, 256])
    x = identity_bottleneck(x, 3, [64, 64, 256])

    x = conv_bottleneck(x, 3, [128, 128, 512])
    x = identity_bottleneck(x, 3, [128, 128, 512])
    x = identity_bottleneck(x, 3, [128, 128, 512])
    x = identity_bottleneck(x, 3, [128, 128, 512])

    x = conv_bottleneck(x, 3, [256, 256, 1024])
    x = identity_bottleneck(x, 3, [256, 256, 1024])
    x = identity_bottleneck(x, 3, [256, 256, 1024])
    x = identity_bottleneck(x, 3, [256, 256, 1024])
    x = identity_bottleneck(x, 3, [256, 256, 1024])
    x = identity_bottleneck(x, 3, [256, 256, 1024])

    x = conv_bottleneck(x, 3, [512, 512, 2048])
    x = identity_bottleneck(x, 3, [512, 512, 2048])
    x = identity_bottleneck(x, 3, [512, 512, 2048])

    
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    return keras.Model(inputs, outputs)
    
if __name__ == '__main__':
    model = make_model(input_shape=(224,224,3), num_classes=1000)
    model.summary()
    keras.utils.plot_model(model, show_shapes=True)