import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, BatchNormalization, Dropout


WEIGHTS_PATH = './pre_trained_weights/vggish_audioset_weights_without_fc2.h5'
WEIGHTS_PATH_TOP = './pre_trained_weights/vggish_audioset_weights.h5'


# from keras import backend as K
## Dropout(rate=0.5, training=True),
# class Dropout(keras.layers.Dropout):
#     """Applies Dropout to the input.
#     Dropout consists in randomly setting
#     a fraction `rate` of input units to 0 at each update during training time,
#     which helps prevent overfitting.
#     # Arguments
#         rate: float between 0 and 1. Fraction of the input units to drop.
#         noise_shape: 1D integer tensor representing the shape of the
#             binary dropout mask that will be multiplied with the input.
#             For instance, if your inputs have shape
#             `(batch_size, timesteps, features)` and
#             you want the dropout mask to be the same for all timesteps,
#             you can use `noise_shape=(batch_size, 1, features)`.
#         seed: A Python integer to use as random seed.
#     # References
#         - [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](
#            http://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf)
#     """
#
#     def __init__(self, rate, training=None, noise_shape=None, seed=None, **kwargs):
#         super(Dropout, self).__init__(rate, noise_shape=None, seed=None, **kwargs)
#         self.training = training
#
#     def call(self, inputs, training=None):
#         if 0. < self.rate < 1.:
#             noise_shape = self._get_noise_shape(inputs)
#
#             def dropped_inputs():
#                 return K.dropout(inputs, self.rate, noise_shape,
#                                  seed=self.seed)
#
#             if not training:
#                 return K.in_train_phase(dropped_inputs, inputs, training=self.training)
#             return K.in_train_phase(dropped_inputs, inputs, training=training)
#         return inputs


class MonteCarloDropout(keras.layers.Dropout):
  def call(self, inputs):
    return super().call(inputs, training=True)

def get_dropout(rate=0.5, mc=True):
    if mc:
        return MonteCarloDropout(rate)
    else:
        return Dropout(rate)


def VGGish(include_top=False, mc=False):
    model = keras.models.Sequential(name="vggish")
    input_shape = (128, 44, 3)
    model.add(Conv2D(64, (3, 3), input_shape=input_shape, strides=(1,1), padding='same', activation='relu', kernel_initializer='glorot_uniform', name='conv1'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool1'))
    model.add(get_dropout(mc=mc))

    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='glorot_uniform', name='conv2'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same',name='pool2'))
    model.add(get_dropout(mc=mc))

    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='glorot_uniform', name='conv3/conv3_1'))
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='glorot_uniform', name='conv3/conv3_2'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same',name='pool3'))
    model.add(get_dropout(mc=mc))

    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='glorot_uniform', name='conv4/conv4_1'))
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='glorot_uniform', name='conv4/conv4_2'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same',name='pool4'))
    model.add(get_dropout(mc=mc))

    # if include_top:
    #     model.load_weights(WEIGHTS_PATH_TOP)
    # else:
    #     model.load_weights(WEIGHTS_PATH)
    return model


if __name__ == '__main__':
    VGGish()