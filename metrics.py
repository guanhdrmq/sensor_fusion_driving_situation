from keras import backend as K

def check_units(y_true, y_pred):
    if y_pred.shape[1] != 1:
      y_pred = y_pred[:,1:2]
      y_true = y_true[:,1:2]
    return y_true, y_pred

def Precision(y_true, y_pred):
    y_true, y_pred = check_units(y_true, y_pred)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def Recall(y_true, y_pred):
    y_true, y_pred = check_units(y_true, y_pred)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def Fmeasure(y_true, y_pred):
    # def precision(y_true, y_pred):
    #     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    #     predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    #     precision = true_positives / (predicted_positives + K.epsilon())
    #     return precision

    # def recall(y_true, y_pred):
    #     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    #     possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    #     recall = true_positives / (possible_positives + K.epsilon())
    #     return recall

    y_true, y_pred = check_units(y_true, y_pred)
    precision = Precision(y_true, y_pred)
    recall = Recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


# def Precision(y_true, y_pred):
#     tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))  # true positives
#     pp = K.sum(K.round(K.clip(y_pred, 0, 1)))  # predicted positives
#     precision = tp / (pp + K.epsilon())
#     return precision
#
# def Recall(y_true, y_pred):
#     tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))  # true positives
#     pp = K.sum(K.round(K.clip(y_true, 0, 1)))  # possible positives
#     recall = tp / (pp + K.epsilon())
#     return recall
#
# def Fmeasure(y_true, y_pred):
#     precision = Precision(y_true, y_pred)
#     recall = Recall(y_true, y_pred)
#     f1 = 2 * ((precision * recall) / (precision + recall + K.epsilon()))
#     return f1