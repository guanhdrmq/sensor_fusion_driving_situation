import os
import cv2
import numpy as np
from keras.utils import np_utils
from VGGish_Model.preprocess_sound import preprocess_sound

config = { "classes": ["safe", "unsafe"]
           }

path_base = "D:/pythonProject/pythonProject/"
left_camera = "two_camera/camera_audio_data/left/"
right_camera = "two_camera/camera_audio_data/right/"
voice_command = "two_camera/camera_audio_data/audio/"

SAMPLE_RATE = 16000

def load_two_camera_data():
    path_left = os.path.join(path_base, left_camera)
    left_images = list(sorted(os.listdir(path_left)))
    left_images = [p for p in left_images if p.endswith('png')]

    path_right = os.path.join(path_base, right_camera)
    right_images = list(sorted(os.listdir(path_right)))
    right_images = [p for p in right_images if p.endswith('png')]

    assert len(left_images) == len(right_images)

    # Containers for the data
    X_left_image = []
    X_right_image = []
    Y = []

    cls2id = {s: i for i, s in enumerate(config['classes'])}
    for i, a in zip(left_images, right_images):
        class_name = i.split('_')[0]
        class_id = cls2id[class_name]

        read_left = cv2.imread(os.path.join(path_left, i), 1)
        read_left = np.resize(read_left, (224, 224, 3))

        read_right = cv2.imread(os.path.join(path_right, a), 1)
        read_right = np.resize(read_right, (224, 224, 3))

        X_left_image.append(read_left)
        X_right_image.append(read_right)

        Y.append(class_id)

    X_left_image = np.array(X_left_image, dtype=np.float32)
    X_right_image = np.array(X_right_image, dtype=np.float32)
    Y = np.array(Y, dtype=np.int)
    Y = np_utils.to_categorical(Y, 2)

    return X_left_image, X_right_image, Y


def load_twocamera_voicecommand():
    path_left = os.path.join(path_base, left_camera)
    left_images = list(sorted(os.listdir(path_left)))
    left_images = [p for p in left_images if p.endswith('png')]

    path_right = os.path.join(path_base, right_camera)
    right_images = list(sorted(os.listdir(path_right)))
    right_images = [p for p in right_images if p.endswith('png')]

    path_voicecommand= os.path.join(path_base,  voice_command)
    audios = list(sorted(os.listdir(path_voicecommand)))
    audios = [p for p in audios if p.endswith('mfcc')]

    assert len(left_images) == len(right_images) == len(audios)

    # Containers for the data
    X_left_image = []
    X_right_image = []
    X_audios = []
    Y = []

    cls2id = {s: i for i, s in enumerate(config['classes'])}
    for i, a, v in zip(left_images, right_images, audios):
        class_name = i.split('_')[0]
        class_id = cls2id[class_name]

        read_left = cv2.imread(os.path.join(path_left, i), 1)
        read_left = np.resize(read_left, (224, 224, 3))

        read_right = cv2.imread(os.path.join(path_right, a), 1)
        read_right = np.resize(read_right, (224, 224, 3))

        mfccs = np.fromfile(os.path.join(path_voicecommand, v))
        #print(mfccs.shape)
        mfccs = mfccs[:1700]
        mfccs = np.resize(mfccs, (496, 64, 1))

        # spectram = preprocess_sound(os.path.join(path_voicecommand, v), SAMPLE_RATE)
        # spectram = np.resize(spectram, (224, 224, 3))

        X_left_image.append(read_left)
        X_right_image.append(read_right)
        X_audios.append(mfccs)
        #X_audios.append(spectram)
        Y.append(class_id)

    X_left_image = np.array(X_left_image, dtype=np.float32)
    X_right_image = np.array(X_right_image, dtype=np.float32)
    X_audios = np.array(X_audios, dtype=np.float32)
    Y = np.array(Y, dtype=np.int)
    Y = np_utils.to_categorical(Y, 2)
    X_audios[np.isinf(X_audios)] = 0
    X_audios[np.isnan(X_audios)] = 0

    return X_left_image, X_right_image, X_audios, Y