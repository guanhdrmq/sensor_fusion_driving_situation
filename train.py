import os
import numpy as np
from matplotlib import pyplot as plt

# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# set_session(tf.Session(config=config))
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from keras.callbacks import ModelCheckpoint, EarlyStopping
checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True,
                             save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=1, mode='auto')

from dataset import load_twocamera_voicecommand
from cnnzoo import merge_camera_voice_command

# load image and audio pairs
(X_left_image, X_right_image, X_audios, Y) = load_twocamera_voicecommand()
# instance of camera and voice_command fusion
model = merge_camera_voice_command()

# collect training history
def show_train_history(train_history, train_metrics, validation_metrics):
    plt.plot(train_history.history[train_metrics])
    plt.plot(train_history.history[validation_metrics])
    plt.title("Train History")
    plt.ylabel(train_metrics)
    plt.xlabel("Epoch")
    plt.legend(["train", "validation"], loc="upper left")

# draw accuracy and loss
def plot(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    show_train_history(history, "acc", "val_acc")
    plt.subplot(1, 2, 2)
    show_train_history(history, "loss", "val_loss")
    plt.show()


if __name__ == '__main__':
    # set training parameters
    nb_epoch = 10
    batch_size = 2
    model.compile(optimizer="adam",
                  loss='categorical_crossentropy',
                  metrics=['accuracy']
                  )

    # training sensor fusion
    history = model.fit([X_left_image, X_right_image, X_audios], Y,
                        epochs=nb_epoch,
                        batch_size=batch_size,
                        validation_split=0.2,
                        verbose=1)

    model.save_weights('my_model_weights.h5')
    # plot training history
    # print(hist.history.keys())
    plot(history)
