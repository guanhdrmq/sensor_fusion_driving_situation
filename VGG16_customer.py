import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, BatchNormalization, Dropout

WEIGHTS_PATH = './pre_trained_weights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
WEIGHTS_PATH_TOP = './pre_trained_weights/vgg16_weights_tf_dim_ordering_tf_kernels.h5'


class MonteCarloDropout(keras.layers.Dropout):
  def call(self, inputs):
    return super().call(inputs, training=True)

def get_dropout(rate=0.5, mc=True):
    if mc:
        return MonteCarloDropout(rate)
    else:
        return Dropout(rate)

def VGG16(include_top=False, mc=False):
    model = keras.models.Sequential(name="vgg16")
    input_shape = (224, 224, 3)
    model.add(keras.layers.Conv2D(64, (3, 3), input_shape=input_shape, padding='same', activation='relu', kernel_initializer='glorot_uniform'))
    model.add(keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(get_dropout(mc=mc))

    model.add(keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(get_dropout(mc=mc))

    model.add(keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(get_dropout(mc=mc))

    model.add(keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(get_dropout(mc=mc))

    model.add(keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(get_dropout(mc=mc))
   # model.summary()

    if include_top:
        model.load_weights(WEIGHTS_PATH_TOP)
    else:
        model.load_weights(WEIGHTS_PATH)
    return model

if __name__ == '__main__':
    VGG16()