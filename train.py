import os
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.model_selection import  KFold

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from keras import backend as K


from keras.callbacks import ModelCheckpoint, EarlyStopping
earlystopping = EarlyStopping(monitor='val_acc', mode="max", min_delta=0, patience=10, verbose=1)

from dataset import load_twocamera_voicecommand
from cnnzoo import merge_camera_voice_command
from metrics import precision,recall,f1


# load image and audio pairs
(X_left_image, X_right_image, X_audios, Y) = load_twocamera_voicecommand()



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

def build_model():
    # 清除之前的模型，省得压满内存
    K.clear_session()
    # instance of camera and voice_command fusion
    model = merge_camera_voice_command()

    model.compile(optimizer="adam",
                  loss='categorical_crossentropy',
                  metrics=['accuracy', precision, recall, f1]
                  )
    return model

if __name__ == '__main__':
    # set training parameters
    nb_epoch = 100
    batch_size = 16
    n_split = 10

    cvscores = []

    model_save_path = 'results'
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)


    kf = KFold(n_splits=n_split, shuffle=True, random_state=None)
    for cnt, (train, test) in enumerate(kf.split(Y)):
        model = build_model()
        print('Training ----------- k fold: {:0>2d} ----------- '.format(cnt))
        checkpoint = ModelCheckpoint(os.path.join(model_save_path, 'model_best_{:0>2d}.h5'.format(cnt)),
                                     monitor='val_acc',
                                     verbose=0,
                                     save_weights_only=False,
                                     save_best_only=True,
                                     mode='auto',
                                     period=1)

        # training sensor fusion
        history = model.fit([X_left_image[train], X_right_image[train], X_audios[train]], Y[train],
                            epochs=nb_epoch,
                            batch_size=batch_size,
                            validation_data=([X_left_image[test], X_right_image[test], X_audios[test]], Y[test]),
                            callbacks=[checkpoint, earlystopping],
                            verbose=2)

        # y_pred = model.predict([X_left_image[test], X_right_image[test], X_audios[test]])
        # matrix = confusion_matrix(Y[test].argmax(axis=1), y_pred.argmax(axis=1))
        # print(matrix)

        # plot training history. check version print(hist.history.keys())
        plot(history)
        plt.show()

        # evaluate the model. socres = [loss, accuracy, precision, recall,f1_score]
        scores = model.evaluate([X_left_image[test], X_right_image[test], X_audios[test]], Y[test], verbose = 0)
        with open('./evaluation.txt', 'a+') as f:
            print('==========Model evaluation k fold: {:0>2d}=========='.format(cnt),file=f)
            print("val_loss:", scores[0],file=f)
            print("val_accuracy:", scores[1],file=f)
            print("val_precision:", scores[2],file=f)
            print("val_recall:", scores[3],file=f)
            print("val_F1:", scores[4],file=f)
            print("====================================================",file=f)


