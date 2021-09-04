from keras.models import Model, Sequential
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D
from keras.engine.topology import get_source_inputs

WEIGHTS_PATH = './pre_trained_weights/vggish_audioset_weights_without_fc2.h5'
WEIGHTS_PATH_TOP = './pre_trained_weights/vggish_audioset_weights.h5'


def VGGish_customer(include_top=False):
    input_shape = (128, 44, 3)
    model = Sequential([
        # Block 1
        Conv2D(64, (3, 3), input_shape=input_shape, strides=(1, 1), padding='same', activation='relu', name='conv1'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool1'),
        # Block 2
        Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv2'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool2'),
        # Block 3
        Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv3/conv3_1'),
        Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv3/conv3_2'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool3'),
        # Block 4
        Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv4/conv4_1'),
        Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv4/conv4_2'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool4'),
    ])

    # if include_top:
    #     model.load_weights(WEIGHTS_PATH_TOP)
    # else:
    #     model.load_weights(WEIGHTS_PATH)
    return model


if __name__ == '__main__':
    VGGish_customer()
