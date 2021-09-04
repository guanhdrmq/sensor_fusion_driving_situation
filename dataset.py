import os
import cv2
import librosa.display
import numpy as np
from keras.utils import np_utils

config = {"classes": ["true", "false"]
          }

# camera and audio data path
path_base = "/Users/Wayne Guan/PycharmProjects/"
left_camera = "camera_audio_fusion_safe/camera_audio_data/left"
right_camera = "camera_audio_fusion_safe/camera_audio_data/right"
voice_command = "camera_audio_fusion_safe/camera_audio_data/audio"

# preprocess audio into spectram
def preproces_audio(path):
    y, sr = librosa.load(path)
    return librosa.feature.melspectrogram(y=y, sr=sr)

# load and read dataset with labels
def load_twocamera_voicecommand():
    path_left = os.path.join(path_base, left_camera)
    left_images = list(sorted(os.listdir(path_left)))
    left_images = [p for p in left_images if p.endswith('png')]

    path_right = os.path.join(path_base, right_camera)
    right_images = list(sorted(os.listdir(path_right)))
    right_images = [p for p in right_images if p.endswith('png')]

    path_voicecommand = os.path.join(path_base, voice_command)
    audios = list(sorted(os.listdir(path_voicecommand)))
    audios = [p for p in audios if p.endswith('wav')]

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

        # mfccs = np.fromfile(os.path.join(path_voicecommand, v))
        # print(mfccs.shape)
        # mfccs = mfccs[:1700]
        # mfccs = np.resize(mfccs, (496, 64, 1))

        spectram = preproces_audio(os.path.join(path_voicecommand, v), )
        # spectram = np.expand_dims(spectram, 3)
        # print(spectram.shape)
        spectram = np.resize(spectram, (128, 44, 3))
        # print(spectram.shape)

        X_left_image.append(read_left)
        X_right_image.append(read_right)
        # X_audios.append(mfccs)
        X_audios.append(spectram)
        Y.append(class_id)

    X_left_image = np.array(X_left_image, dtype=np.float32)
    X_right_image = np.array(X_right_image, dtype=np.float32)
    X_audios = np.array(X_audios, dtype=np.float32)
    Y = np.array(Y, dtype=np.int)
    Y = np_utils.to_categorical(Y, 2)
    X_audios[np.isinf(X_audios)] = 0
    X_audios[np.isnan(X_audios)] = 0

    return X_left_image, X_right_image, X_audios, Y


if __name__ == '__main__':
    load_twocamera_voicecommand()
