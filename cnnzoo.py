from keras.applications import VGG16
from keras.models import Model, Sequential
from keras import regularizers
from keras.layers import Input, Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization, \
    Concatenate
from keras import models, layers
from vggish_customer import VGGish_customer

WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1' \
               '/vgg16_weights_tf_dim_ordering_tf_kernels.h5 '
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1' \
                      '/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5 '


def audio_extractor():
    conv_audio = VGGish_customer(include_top=False)
    model = Sequential()
    model.add(conv_audio)
    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.summary()
    return model

def vgg16_extractor():
    conv_base = VGG16(weights='./pre_trained_weights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                      include_top=False,
                      input_shape=(224, 224, 3))

    model = Sequential()
    model.add(conv_base)
    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    # model.summary()
    return model

def merge_camera_voice_command():
    CNN_left = vgg16_extractor()
    CNN_right = vgg16_extractor()
    CNN_audio = audio_extractor()

    for layer in CNN_left.get_layer("vgg16").layers:
        layer._name = "left_" + layer.name
    for layer in CNN_right.get_layer("vgg16").layers:
        layer._name = "left_" + layer.name

    for layer in CNN_left.layers:
        layer.name = "left_" + layer.name
    for layer in CNN_right.layers:
        layer.name = "right_" + layer.name
    for layer in CNN_audio.layers:
        layer.name = "audio_" + layer.name

    CNN_left.get_layer("vgg16_input").name = "left_vgg16_input"
    CNN_right.get_layer("vgg16_input").name = "right_vgg16_input"

    input_left = CNN_left.input
    input_right = CNN_right.input
    input_audio = CNN_audio.input

    print(input_left.shape, input_right.shape, input_audio.shape)
    print(CNN_left.output, CNN_right.output, CNN_audio.output)

    merged = Concatenate(axis=-1)([CNN_left.output, CNN_right.output, CNN_audio.output])
    merged = BatchNormalization()(merged)
    merged = Dropout(0.5)(merged)
    fc = Dense(512, activation='relu')(merged)
    fc = Dropout(0.5)(fc)
    out = Dense(2, activation='softmax', name='predictions')(fc)

    # Return the model object
    model = Model(inputs=[input_left, input_right, input_audio], outputs=out)
    return model


if __name__ == '__main__':
    merge_camera_voice_command()
