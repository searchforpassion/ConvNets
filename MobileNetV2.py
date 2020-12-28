from tensorflow import keras
from tensorflow.keras import layers

def downsample_bottleneck(inputs, stride, filters):
    
    x = layers.Conv2D(filters, kernel_size=1, padding='same', use_bias=False, activation=None)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6.)(x)
    
    x = layers.ZeroPadding2D(padding=(1, 1))(x)
    x = layers.DepthwiseConv2D(kernel_size=3, strides=stride, activation=None, use_bias=False, padding='valid')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6.)(x)

    x = layers.Conv2D(filters, kernel_size=1, padding='same', use_bias=False, activation=None)(x)
    x = layers.BatchNormalization()(x)

    return x

def inverted_res_bottleneck(inputs, stride, filters):
    
    x = layers.Conv2D(filters, kernel_size=1, padding='same', use_bias=False, activation=None)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6.)(x)
    
    x = layers.DepthwiseConv2D(kernel_size=3, strides=stride, activation=None, use_bias=False, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6.)(x)

    x = layers.Conv2D(filters, kernel_size=1, padding='same', use_bias=False, activation=None)(x)
    x = layers.BatchNormalization()(x)
    
    return layers.add([x, inputs])

def make_model(input_shape=(224,224,3), num_classes=1000):
    inputs = keras.Input(shape=input_shape)
    x = layers.ZeroPadding2D(padding=(1, 1))(inputs)
    x = layers.Conv2D(32, kernel_size=3, strides=(2, 2), padding='valid', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6.)(x)

    x = downsample_bottleneck(x, filters=16, stride=1)

    x = downsample_bottleneck(x, filters=24, stride=2)
    x = inverted_res_bottleneck(x, filters=24, stride=1)

    x = downsample_bottleneck(x, filters=32, stride=2)
    x = inverted_res_bottleneck(x, filters=32, stride=1)
    x = inverted_res_bottleneck(x, filters=32, stride=1)

    x = downsample_bottleneck(x, filters=64, stride=2)
    x = inverted_res_bottleneck(x, filters=64, stride=1)
    x = inverted_res_bottleneck(x, filters=64, stride=1)
    x = inverted_res_bottleneck(x, filters=64, stride=1)

    x = downsample_bottleneck(x, filters=96, stride=1)
    x = inverted_res_bottleneck(x, filters=96, stride=1)
    x = inverted_res_bottleneck(x, filters=96, stride=1)

    x = downsample_bottleneck(x, filters=160, stride=2)
    x = inverted_res_bottleneck(x, filters=160, stride=1)
    x = inverted_res_bottleneck(x, filters=160, stride=1)

    x = downsample_bottleneck(x, filters=320, stride=1)

    x = layers.Conv2D(1280, kernel_size=1, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6.)(x)
    
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    return keras.Model(inputs, outputs)
    
if __name__ == '__main__':
    model = make_model(input_shape=(224,224,3), num_classes=1000)
    model.summary()
    keras.utils.plot_model(model, show_shapes=True)