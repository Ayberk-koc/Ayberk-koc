import pandas as pd
import numpy as np
from scikeras.wrappers import KerasClassifier
from sklearn.utils.class_weight import compute_class_weight
from Ma_revamp.bio.plot_tools import plot_confusion_matr
from Ma_revamp.bio.diverse_tools import remove_outliers_iqr_by_class
from Ma_revamp.total.shap_calculation import calc_and_save_shap_values
import os
from Ma_revamp.bio.plot_tools import class_balance_plot


# bio_path_filtered = r"C:\Users\ayber\Desktop\BioP_ev_test\openNP_gut_checkpoint\tot_data(aus_filtered_mit_lvl_info)\K_0-10kHz_100.csv"
bio_path_filtered = r"C:\Users\ayber\Desktop\BioP_ev_test\openNP_gut_checkpoint\tot_data(aus_filtered_mit_lvl_info)\D_0-10kHz_100.csv"
# bio_path_filtered = r"C:\Users\ayber\Desktop\pelt_test\Bio_R\bio_R.csv"
data = pd.read_csv(bio_path_filtered)
classes = ["a3", "a4", "a5"]
classes_nr = len(classes)

data = remove_outliers_iqr_by_class(data, bound=1, groupby_label="label")



# wenn ich binary classification mache, dann kommentieren das hier aus!
y = pd.get_dummies(data["label"], prefix="label")
data = data.drop(columns=["label"])
data = pd.concat([data, y], axis=1)



x = data.iloc[:, [0, 1, 2, 3, 4]].values          # ,3  (nr_lvl raus genommen)
y = data.iloc[:, -classes_nr:].values



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
# scaler = MinMaxScaler(feature_range=(0, 500))
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)           #bei pipeline check das scaling aus machen, denn das ist ja das was geprüft wird
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)


## hier undersampling / oversmapling

from imblearn.over_sampling import SMOTE, RandomOverSampler
# oversampler = SMOTE()
oversampler = RandomOverSampler()

from imblearn.under_sampling import RandomUnderSampler
undersampler = RandomUnderSampler()


x_train, y_train = oversampler.fit_resample(x_train, y_train)         #mache auch wenn ich grid search mache
x_train, y_train = undersampler.fit_resample(x_train, y_train)


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

def create_model(hidden_unit_size, activation_func, regularization):                #!!!!!!!! mache bisschenr egularization rein!!!!!!! -> ich sehe schon overfitting im vgl von val zu train data!!!!!!
    ann = tf.keras.models.Sequential()                                              #!!!!!!!! und noch grid search für layer anzahl!!!!!
    ann.add(tf.keras.layers.InputLayer(input_shape=(x.shape[1],)))      #das x.shape[1] ist für die auto wahl von input layern
    for hidden_unit in hidden_unit_size:
       ann.add(tf.keras.layers.Dense(units=hidden_unit, activation=activation_func, activity_regularizer=tf.keras.regularizers.l2(regularization)))
       ann.add(tf.keras.layers.Dropout(0.01))   #vorher hier 0.01
    ann.add(tf.keras.layers.Dense(units=classes_nr, activation="softmax"))
    return ann


# mache model einmal nicht dynamisch, sodner von oben nach unten. Kann so regularization einstellen.
# mache einmal regularization ans ende. Auch mache einen andere learning rate vom optimizer

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(np.argmax(y_train, axis=1)), y=np.argmax(y_train, axis=1))
class_weights_dict = dict(enumerate(class_weights))

from keras import optimizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
adam = optimizers.Adam(lr=0.0001, beta_1=0.95, beta_2=0.999) #, decay=0.0001
early_stop = EarlyStopping(monitor='val_loss', patience=5)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1)

import tensorflow as tf
new_model = True
if new_model:
    model = KerasClassifier(model=create_model, loss="categorical_crossentropy", optimizer=adam,       # optimizer="adam", # optimizer__optimizer_config = adam_config,
                           metrics=["accuracy", f1], batch_size=15, epochs=100,
                           model__hidden_unit_size=[90, 90, 90], # optimizer__lr=0.0005,
                           model__activation_func="relu", model__regularization=0.00005               #0.00005
                            )
    model.fit(x_train, y_train, validation_data=(x_val, y_val), class_weight=class_weights_dict,                        # , class_weight=class_weights_dict
              callbacks=[early_stop, reduce_lr])          # , reduce_lr                                                               #das class_weight wird an fit method des ann's gegeben. Doch brauche keine class weight wenn ich under/oversample
    model.model_.save(r"C:\Users\ayber\Desktop\Deep_models\bio\ann_tot_bio.h5")                                      #das class_weight brauche ich nicht wenn ich over/undersample

    ########## ohne scikeras ##########
    # model = create_model(None, [9]*4, "relu")
    # model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    # model.fit(x_train, y_train, batch_size=32, epochs=100, class_weight=class_weights_dict)
else:
    ann = tf.keras.models.load_model(r"C:\Users\ayber\Desktop\Deep_models\bio\ann_tot_bio.h5")
    model = KerasClassifier(model=ann)
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
save_path = r"C:\Users\ayber\Desktop\conf_mat_arrs\bio_multi\annD.npy"
np.save(save_path, cm)
##### conf mat erg speichern


exit()
# ################ shap analyse ##################
exp_folder_name = "bio_new"
folder_path = rf"C:\Users\ayber\Desktop\shap_value_ordner\{exp_folder_name}"
filename = "ann_shap_values_test_K_new"
if not os.path.isdir(folder_path):
    os.mkdir(folder_path)

calc_and_save_shap_values(model.model_, x_train, folder_path, filename=filename)
# ################ shap analyse ##################






