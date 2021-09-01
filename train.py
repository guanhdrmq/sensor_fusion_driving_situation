import os
from matplotlib import pyplot as plt

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

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

# set training parameters
nb_epoch = 300
batch_size = 8
model.compile(optimizer="adam",
              loss='categorical_crossentropy',
              metrics=['accuracy']
              )

# 训练 verbose：日志显示,verbose = 0 为不在标准输出流输出日志信息, verbose = 1 为输出进度条记录, verbose = 2 为每个epoch输出一行记录
hist = model.fit([X_left_image, X_right_image, X_audios], Y,
                 epochs=nb_epoch,
                 batch_size=batch_size,
                 validation_split=0.25,
                 verbose=1)

model.save_weights('my_model_weights.h5')

# plot training history
# print(hist.history.keys())
plt.plot(hist.history["acc"])
plt.plot(hist.history['val_acc'])
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title("model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy", "Validation Accuracy", "loss", "Validation Loss"])
fig = plt.figure()
fig.savefig('performance.png')
plt.show()







