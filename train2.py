import os
import logging
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold, train_test_split

# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# set_session(tf.Session(config=config))
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from keras.callbacks import ModelCheckpoint, EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=2)
try:
    from keras.layers import K
except:
    import keras.backend as K

from dataset import load_twocamera_voicecommand
from cnnzoo import merge_camera_voice_command
from metrics import Metrics

model_save_path = './weights.hdf5'


def get_logger(path, mode='a'):
    """
    Create a basic logger
    :param path: log file path
    :param mode: 'a' for append, 'w' for write
    :return: a logger with log level INFO
    """
    name, _ = os.path.splitext(os.path.basename(path))
    logging.basicConfig(format="%(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(level=logging.INFO)
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    handler = logging.FileHandler(path, mode=mode)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger.info


logger = get_logger('./train.log', mode='w')

nb_epoch = 10
batch_size = 1


def train(model, X_left_image, X_right_image, X_audios, Y, indices, model_save_path):
    model.compile(optimizer="adam",
                  loss='categorical_crossentropy',
                  metrics=['accuracy']
                  )

    val_accs = []
    n_split = 10
    kf = KFold(n_splits=n_split, shuffle=True, random_state=None)
    for cnt, (train, test) in enumerate(kf.split(indices)):
        checkpoint = ModelCheckpoint(os.path.join(model_save_path, 'model_best_{:0>2d}.h5'.format(cnt)),
                                     monitor='val_acc',
                                     verbose=0,
                                     save_weights_only=False,
                                     save_best_only=True,
                                     mode='max',
                                     period=0)
        logger('Training ----------- k fold: {:0>2d} ----------- '.format(cnt))
        history = model.fit([X_left_image[train], X_right_image[train], X_audios[train]], Y[train],
                            epochs=nb_epoch,
                            batch_size=batch_size,
                            validation_data=([X_left_image[test], X_right_image[test], X_audios[test]], Y[test]),
                            callbacks=[checkpoint, Metrics(valid_data=([X_left_image[test], X_right_image[test],
                                                                        X_audios[test]], Y[test]))],
                            verbose=1)
        val_accs.append(np.array(history.history['val_acc']).max())
        labs = []
        scores = []
        for k, v in history.history.items():
            labs.append(k)
            scores.append(v)
            value = (k, np.array(v).max())
            logger(value)
        [plt.plot(scores[i], label=labs[i]) for i in range(len(scores)) if 'acc' in labs[i]]
        plt.title('train accuracy vs. val accuracy')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(model_save_path, '{:0>2d}_accuracy.png'.format(cnt)))
        plt.close()

        [plt.plot(scores[i], label=labs[i]) for i in range(len(scores)) if 'loss' in labs[i]]
        plt.title('train loss vs. val loss')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(model_save_path, '{:0>2d}_loss.png'.format(cnt)))
        plt.close()

        [plt.plot(scores[i], label=labs[i]) for i in range(len(scores)) if
         'acc' not in labs[i] and 'loss' not in labs[i]]
        plt.title('Validation Precision & Recall & F1 score')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(model_save_path, '{:0>2d}_eval.png'.format(cnt)))
        plt.close()

    val_accs_mean = np.array(val_accs).mean()
    logger('-------------------------------------------------------------')
    logger('validation accuracies on {} fold'.format(n_split))
    logger(val_accs)
    logger('mean validation accuracy: {:.6f}'.format(val_accs_mean))
    logger('---------------------------DONE------------------------------')
    return val_accs_mean


if __name__ == '__main__':
    # load image and audio pairs
    X_left_image, X_right_image, X_audios, Y = load_twocamera_voicecommand()
    # instance of camera and voice_command fusion
    model = merge_camera_voice_command()
    # creating weights path
    model_save_path = 'results'
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    # loop data
    indices = [i for i in range(len(X_left_image))]
    train(model, X_left_image, X_right_image, X_audios, Y, indices, model_save_path)
