import pandas as pd
import numpy as np
from Ma_revamp.bio.plot_tools import plot_confusion_matr
from Ma_revamp.bio.plot_tools import class_balance_plot
from Ma_revamp.bio.diverse_tools import remove_outliers_iqr_by_class
from Ma_revamp.total.shap_calculation import calc_and_save_shap_values
import os


bio_path_filtered = r"C:\Users\ayber\Desktop\BioP_ev_test\openNP_gut_checkpoint\tot_data(aus_filtered_mit_lvl_info)\R_0-10kHz_100.csv"
data = pd.read_csv(bio_path_filtered)
classes = ["a3", "a4", "a5"]

data = remove_outliers_iqr_by_class(data, bound=1, groupby_label="label")           #das muss vor entfernen der clas gecallt werden



x = data.iloc[:, [0, 1, 2, 3, 4]].values        #mit std
y = data.iloc[:, -1].values

class_balance_plot(y, classes)
#########################################
rem = np.where(x[:, 0] >= 0.001)     #hier entferne alle mit zu kleiner dwell time
x = x[rem]
y = y[rem]
class_balance_plot(y, classes)
#########################################



from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)



from imblearn.over_sampling import RandomOverSampler, SMOTE
# oversampler = SMOTE()
oversampler = RandomOverSampler()   #sampling_strategy={0: 2500, 1: 2500, 2: 2500}


from imblearn.under_sampling import RandomUnderSampler
undersampler = RandomUnderSampler() #sampling_strategy={1: 2500, 2: 2500}


x_train, y_train = undersampler.fit_resample(x_train, y_train)
x_train, y_train = oversampler.fit_resample(x_train, y_train)

class_balance_plot(y_train, classes)



# from sklearn.ensemble import RandomForestClassifier
# rfc = RandomForestClassifier(n_estimators=70, criterion="log_loss") #n_estimators=70, criterion="entropy"    #n_estimators=10, criterion="entropy"
# rfc.fit(x_train, y_train)
# y_pred = rfc.predict(x_test)
# print(rfc.classes_)


from xgboost.sklearn import XGBClassifier
xgb = XGBClassifier()   #n_estimators=80
xgb.fit(x_train, y_train)
y_pred = xgb.predict(x_test)
print(xgb.classes_)



accuracy = plot_confusion_matr(y_test, y_pred, classes)
print("accuracy: ", accuracy)
from sklearn.metrics import f1_score
f1_weighted = f1_score(y_test, y_pred, average='weighted')
f1_macro = f1_score(y_test, y_pred, average='macro')
f1_micro = f1_score(y_test, y_pred, average='micro')

print("f1_weighted: ", f1_weighted)
print("f1_macro: ", f1_macro)
print("f1_micro: ", f1_micro)



# ###### hier um conf mat ergebnis zu speicher
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_test, y_pred, normalize="true")
# save_path = r"C:\Users\ayber\Desktop\conf_mat_arrs\bio\xgb.npy"
# np.save(save_path, cm)
# ##### conf mat erg speichern




exit()
################ shap analyse ##################
exp_folder_name = "bio_new"
folder_path = rf"C:\Users\ayber\Desktop\shap_value_ordner\{exp_folder_name}"
filename = "rfc_shap_values_test_R"
if not os.path.isdir(folder_path):
    os.mkdir(folder_path)

calc_and_save_shap_values(rfc, x_train, folder_path, filename=filename)
################ shap analyse ##################





