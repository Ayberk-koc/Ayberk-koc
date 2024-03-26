from yellowbrick.classifier import ClassificationReport, ConfusionMatrix, ROCAUC
import pandas as pd
import numpy as np
from matplotlib.cm import ScalarMappable
from scikeras.wrappers import KerasClassifier
from sklearn.utils.class_weight import compute_class_weight
from Ma_revamp.bio.plot_tools import plot_confusion_matr
import os
from Ma_revamp.bio.diverse_tools import remove_outliers_iqr_by_class
from Ma_revamp.total.shap_calculation import calc_and_save_shap_values
from sklearn.inspection import permutation_importance
from Ma_revamp.bio.plot_tools import bar_plot
from Ma_revamp.bio.plot_tools import class_balance_plot

bio_path_filtered = r"C:\Users\ayber\Desktop\BioP_ev_test\openNP_gut_checkpoint\tot_data(aus_filtered_mit_lvl_info)\K_0-10kHz_100.csv"
data = pd.read_csv(bio_path_filtered)
classes = ["a3", "a4", "a5"]
classes_nr = len(classes)

data = remove_outliers_iqr_by_class(data, bound=1, groupby_label="label")


# wenn ich binary classification mache, dann kommentieren das hier aus!
y = pd.get_dummies(data["label"], prefix="label")
total_data = data.drop(columns=["label"])
total_data = pd.concat([total_data, y], axis=1)



x = total_data.iloc[:, [0, 1, 2, 3, 4]].values
y = total_data.iloc[:, -classes_nr:].values


class_balance_plot(np.argmax(y, axis=1), classes)
#########################################
rem = np.where(x[:, 0] >= 0.001)     #hier entferne alle mit zu kleiner dwell time
x = x[rem]
y = y[rem]
class_balance_plot(np.argmax(y, axis=1), classes)



from sklearn.model_selection import train_test_split
x_trainval, x_test, y_trainval, y_test = train_test_split(x, y, test_size=0.2)
x_train, x_val, y_train, y_val = train_test_split(x_trainval, y_trainval, test_size=0.2)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# scaler = MinMaxScaler(feature_range=(0, 255))
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)           #bei pipeline check das scaling aus machen, denn das ist ja das was geprüft wird
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)


from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import TomekLinks, RandomUnderSampler
oversampler = RandomOverSampler()                               #sampling_strategy={0: 300, 1: 300, 2: 300}
undersampler = RandomUnderSampler()                         #sampling_strategy={0: 300, 1: 300, 2: 300}

x_train, y_train = oversampler.fit_resample(x_train, y_train)
x_train, y_train = undersampler.fit_resample(x_train, y_train)

#das für das cnn input für shap
x_train_pre_shape = np.copy(x_train)


featnr = x_train.shape[1]
train_length = x_train.shape[0]
test_length = x_test.shape[0]
val_length = x_val.shape[0]

x_train = x_train.reshape((train_length, 1, featnr, 1))
x_test = x_test.reshape((test_length, 1, featnr, 1))
x_val = x_val.reshape((val_length, 1, featnr, 1))



############# hier experimentell ######
import keras.backend as K
def f1(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val
###############################



import tensorflow as tf

def create_model(conv_layers, hidden_unit_size, conv_regularization, activation_func):
    reg = tf.keras.regularizers.l2(conv_regularization)
    cnn = tf.keras.models.Sequential()
    cnn.add(tf.keras.layers.InputLayer(input_shape=(1, featnr, 1)))
    cnn.add(tf.keras.layers.Conv1D(conv_layers[0], kernel_size=3, strides=1, activation=activation_func, activity_regularizer=reg))
    # cnn.add(tf.keras.layers.MaxPooling1D(2))
    #cnn.add(tf.keras.layers.Conv1D(conv_layers[1], kernel_size=3, activation=activation_func, activity_regularizer=reg))   #MAYBE HIERMIT MAL VERSUCHEN!!
    # cnn.add(tf.keras.layers.MaxPooling1D(2))
    cnn.add(tf.keras.layers.Flatten())
    for hidden_layer in hidden_unit_size:
        cnn.add(tf.keras.layers.Dense(hidden_layer, activation=activation_func, activity_regularizer=reg))
        cnn.add(tf.keras.layers.Dropout(0.05))
    cnn.add(tf.keras.layers.Dense(classes_nr, activation='softmax'))
    return cnn

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(np.argmax(y_train, axis=1)), y=np.argmax(y_train, axis=1))
class_weights_dict = dict(enumerate(class_weights))

from keras import optimizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
adam = optimizers.Adam(lr=0.0001, beta_1=0.95, beta_2=0.999) #, decay=0.0001
early_stop = EarlyStopping(monitor='val_loss', patience=5)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1)

new_model = True
if new_model:
    model = KerasClassifier(model=create_model, batch_size=15, epochs=100, loss="categorical_crossentropy", optimizer=adam,
                            metrics=["accuracy", f1], model__conv_layers=[30,], model__hidden_unit_size=(30, 30),   #hier vorher überall 90
                            model__conv_regularization=0.0001, model__activation_func="relu",       # optimizer__lr=0.1,
                            )
    model.fit(x_train, y_train, validation_data=(x_val, y_val), class_weight=class_weights_dict,
              callbacks=[early_stop, reduce_lr])
    # model.model_.save(r"C:\Users\ayber\Desktop\Deep_models\bio\cnn_bio_exp.h5")
else:
    model = tf.keras.models.load_model(r"C:\Users\ayber\Desktop\Deep_models\total\cnn_bio_exp.h5")
    model = KerasClassifier(model)
    model.initialize(x_train, y_train)

y_pred = model.predict(x_test)

accuracy = plot_confusion_matr(y_test, y_pred, classes)
from sklearn.metrics import f1_score
f1_weighted = f1_score(y_test, y_pred, average='weighted')  #skalar, f1_score gewichtet zum average
f1_macro = f1_score(y_test, y_pred, average='macro')        #skaler, f1_score ungewichtet zum average
f1_micro = f1_score(y_test, y_pred, average='micro')        #skalar, total f1_score (kein average, einfach nur TP, TN etc. gez#hlt)

print("accuracy: ", accuracy)
print("f1_weighted: ", f1_weighted)
print("f1_macro: ", f1_macro)
print("f1_micro: ", f1_micro)


###### hier um conf mat ergebnis zu speicher
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1), normalize="true")
save_path = r"C:\Users\ayber\Desktop\conf_mat_arrs\bio\cnn.npy"
np.save(save_path, cm)
##### conf mat erg speichern


exit()
########## ändere diesen folder path, mache einfach hardcode
exp_folder_name = "bio_new"
folder_path = rf"C:\Users\ayber\Desktop\shap_value_ordner\{exp_folder_name}"
filename = "cnn_shap_values_test_R"
if not os.path.isdir(folder_path):
    os.mkdir(folder_path)

calc_and_save_shap_values(model.model_, x_train_pre_shape, folder_path, filename=filename)






