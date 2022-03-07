import os, random
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
import numpy as np

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from keras.callbacks import ModelCheckpoint, EarlyStopping
earlystopping = EarlyStopping(monitor='val_acc', mode="max", min_delta=0, patience=10, verbose=1)

# reproduce the same results
seed_value =0
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.set_random_seed(seed_value)
from keras import backend as K
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

from dataset import load_twocamera_voicecommand
from cnnzoo import merge_camera_voice_command
from metrics import Precision,Recall,Fmeasure
from reliability_diagrams import *


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
    plt.savefig("images/loss_accuracy_{:0>2d}.png".format(cnt))
    plt.show()

# build model and clear session
def build_model():
    # release memory
    K.clear_session()
    # instance of camera and voice_command fusion
    model = merge_camera_voice_command()


    model.compile(optimizer="adam",
                  loss='categorical_crossentropy',
                  #loss=triplet_loss_adapted_from_tf,
                  metrics=['accuracy',Precision, Recall, Fmeasure]
                  )
    return model

# Monte Carlo prediction and uncertainty
def predict_with_uncertainty(model, X_left_image, X_right_image, X_audios, batch_size=16, num_iterations=10):
    last_layer = model.layers[-1]
    results = np.zeros((num_iterations,
                        X_audios.shape[0],
                        last_layer.output_shape[1]),
                        dtype="float")
    for i in range(num_iterations):
        results[i] = model.predict([X_left_image, X_right_image, X_audios], batch_size=batch_size)

    predictions = results.mean(axis=0)
    uncertainty = results.std(axis=0)
    return predictions, uncertainty

# Monte Carlo Accuracy
def MCdropout_Accuracy(model, X_left_image, X_right_image, X_audios, Y, batch_size=16):
    import tqdm
    mc_predictions = []
    for i in tqdm.tqdm(range(100)):
        y_p = model.predict([X_left_image, X_right_image, X_audios], batch_size=batch_size)
        mc_predictions.append(y_p)

    from sklearn.metrics import accuracy_score
    accs = []
    for y_p in mc_predictions:
        acc = accuracy_score(Y.argmax(axis=1), y_p.argmax(axis=1))
        accs.append(acc)
    MC_accuracy = (sum(accs) / len(accs))
    print("MC accuracy: {:.1%}".format(MC_accuracy))

    mc_ensemble_pred = np.array(mc_predictions).mean(axis=0).argmax(axis=1)
    ensemble_acc = accuracy_score(Y.argmax(axis=1), mc_ensemble_pred)
    print("MC-ensemble-accuracy: {:.1%}".format(ensemble_acc))

    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 15,
             }

    plt.xlabel("Accuracy", font2)
    plt.ylabel("Count", font2)
    plt.hist(accs)
    plt.axvline(x=ensemble_acc, color="r")
    plt.savefig("images/MC_accuracy_distribution_{:0>2d}.png".format(cnt))

    return sum(accs)/len(accs), ensemble_acc


from tqdm import tqdm
def predict_with_uncertainty_MCdropout(model,
                             X_left_image, X_right_image, X_audios, Y,
                             num_iterations=100):
    last_layer = model.layers[-1]
    results = np.zeros((num_iterations,
                        X_audios.shape[0],
                        last_layer.output_shape[1]),
                        dtype="float")
    for i in tqdm(range(num_iterations)):
        pred = model.predict([X_left_image, X_right_image, X_audios])
        np.save("temp/pred_{}".format(i), pred)
        results[i] = pred

    predictions = results.mean(axis=0)

    np.save("results",results)
    np.save("predictions",predictions)

    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(Y, axis=1)

    accs = []
    for i in range(len(results)):
        accs.append(sum(np.argmax(results[i], axis=1) == y_true) / len(y_true))

    rangemin = int(min(accs)*400)
    rangemax = int(max(accs)*400)
    lenth = rangemax-rangemin
    rangemin -= int((10-lenth)/2)
    rangemax += 10-lenth-(int((10-lenth)/2))

    bins = [i/400 for i in range(rangemin,rangemax)]
    acc_bin = np.zeros(10)
    for acc in accs:
        acc_bin[bins.index(acc)] += 1

    maxacc = max(acc_bin)//10*10+10

    plt.style.use('bmh')

    plt.figure(figsize=(5,5))

    bar1 = plt.bar([i/400 for i in range(rangemin,rangemax)],acc_bin, alpha=0.7, width=0.0025, linewidth=1.5, edgecolor='black')
    plt.vlines(sum(y_pred == y_true) / len(y_pred),0,maxacc,'r')

    plt.xticks([i/200 for i in range(int(rangemin/2),int(rangemax/2)+1)])

    plt.xlim(bins[0],bins[-1])
    plt.ylim(0,maxacc)

    plt.xlabel("Accuracy")
    plt.ylabel("Count")

    plt.savefig("images/MC_distribution_{:0>2d}.png".format(cnt))

if __name__ == '__main__':
    # set training parameters
    nb_epoch = 100
    batch_size = 16
    n_split = 10
    cvscores = []

    model_save_path = 'results'
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    model_loss_accuracy_distribution_save_path = 'images'
    if not os.path.exists(model_loss_accuracy_distribution_save_path):
        os.makedirs(model_loss_accuracy_distribution_save_path)

    kf = KFold(n_splits=n_split, shuffle=True, random_state=seed_value)
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
        # if os.path.exists('results'):
        #     model.load_weights('results/model_best_{:0>2d}.h5'.format(cnt))
             # if successful looded, print out following message
          #  print("^^^^^^^^^^^^^^^^^^^^^^checkpoint_loaded^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

        # train sensor fusion
        history = model.fit([X_left_image[train], X_right_image[train], X_audios[train]], Y[train],
                                epochs=nb_epoch,
                                batch_size=batch_size,
                                shuffle=True,
                                validation_data=([X_left_image[test], X_right_image[test], X_audios[test]], Y[test]),
                                callbacks=[checkpoint, earlystopping],
                                verbose=2)

        # plot training history. check version print(hist .history.keys())
        plot(history)
        plt.show()
        # model.summary()

        #########################################################################
        # score of the normal model. socres = [loss, accuracy, precision, recall,f1_score]
        scores = model.evaluate([X_left_image[test], X_right_image[test], X_audios[test]], Y[test], verbose=0)

        # score of the mc model
        mc_accuracy, mc_ensemble_accuracy = MCdropout_Accuracy(model, X_left_image[test], X_right_image[test], X_audios[test], Y[test])
       # predict_with_uncertainty_MCdropout(model, X_left_image[test], X_right_image[test], X_audios[test], Y[test])

        ##########################################################################
        # Calibration error
        test_preds = model.predict([X_left_image[test], X_right_image[test], X_audios[test]])
        y_confs = np.max(test_preds, axis=1)
        y_preds = np.argmax(test_preds, axis=1)
        y_classes = np.argmax(Y[test], axis=1)

        from keras_uncertainty.utils import classifier_calibration_error
        calib_error = classifier_calibration_error(y_preds, y_classes, y_confs, num_bins=10)
        print('Calibration Error:', calib_error)

        # histogram calibration error
        fig = reliability_diagram(y_classes, y_preds, y_confs, num_bins=10, draw_ece=True,
                                  draw_bin_importance="alpha", draw_averages=True,
                                  title="reliability_diagrams", figsize=(6, 6), dpi=100,
                                  return_fig=True)

        # import uncertainty_metrics.numpy as um
        # ece = um.ece(y_classes, y_preds, num_bins=10)
        # diagram = um.reliability_diagram(y_classes, y_preds)


        ##########################################################################

        with open('evaluation905.txt', 'a+') as f:
            print('==========Model evaluation k fold: {:0>2d}=========='.format(cnt),file=f)
            print("val_loss:", scores[0],file=f)
            print("val_accuracy:", scores[1],file=f)
            print("val_precision:", scores[2],file=f)
            print("val_recall:", scores[3],file=f)
            print("val_F1:", scores[4],file=f)
            print("MCdropout Accuracy and MC-ensemble-accuracy", mc_accuracy, mc_ensemble_accuracy, file=f)
            print("Calibration Error:", calib_error ,file=f)
            print("======================== ============================",file=f)

