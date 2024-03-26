import numpy as np
import pandas as pd
from Ma_revamp.bio.plot_tools import plot_confusion_matr
from Ma_revamp.bio.diverse_tools import remove_outliers_iqr_by_class
import os


from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler

class CustomClassifier(BaseEstimator, ClassifierMixin):

    @staticmethod
    def transform_arr_to_proba(arr1, arr2):
        to_app = []
        mask = arr1[:, 0] < 0.5
        j = 0
        for i in range(len(mask)):
            if mask[i]:
                row_to_append = arr2[j]
                j += 1
            else:
                row_to_append = np.ones(shape=(arr2.shape[1],))
            to_app.append(row_to_append)
        arr2 = np.array(to_app)

        vals = []
        for i in range(arr1.shape[0]):
            row = arr1[i]
            row2 = (arr2[i] / arr2.shape[1]) * row[1]
            row_to_append = np.concatenate((row[:-1], row2))
            vals.append(list(row_to_append))
        arr = np.array(vals)
        return arr


    def __init__(self, binary_clf, multiclass_clf):
        self.binary_clf = binary_clf
        self.multiclass_clf = multiclass_clf
        self.binary_scaler = StandardScaler()
        self.multiclass_scaler = StandardScaler()

    def fit(self, X, y):
        X_binary_scaled = self.binary_scaler.fit_transform(X)
        y_binary = np.where(y != 0, 1, y)
        self.binary_clf.fit(X_binary_scaled, y_binary)
        binary_predictions = self.binary_clf.predict(X_binary_scaled)

        X_multiclass = X[binary_predictions == 1]
        y_multiclass = y[binary_predictions == 1]
        # Perform feature scaling for multiclass classification
        X_multiclass_scaled = self.multiclass_scaler.fit_transform(X_multiclass)
        # Perform multiclass classification
        self.multiclass_clf.fit(X_multiclass_scaled, y_multiclass)
        return self

    def predict(self, X):
        X_binary_scaled = self.binary_scaler.transform(X)
        binary_predictions = self.binary_clf.predict(X_binary_scaled)

        X_multiclass = X[binary_predictions == 1]
        X_multiclass_scaled = self.multiclass_scaler.transform(X_multiclass)
        multiclass_predictions = self.multiclass_clf.predict(X_multiclass_scaled)
        binary_predictions[binary_predictions == 1] = multiclass_predictions
        return binary_predictions   #hier sind alle predictions nun drinne obwohl es binary predictions heißt! -> das ändere noch safe!!!

    def predict_proba(self, X):
        X_binary_scaled = self.binary_scaler.transform(X)
        binary_predictions = self.binary_clf.predict_proba(X_binary_scaled)

        bin_pred_mask = np.argmax(binary_predictions, axis=1)

        X_multiclass = X[bin_pred_mask == 1]
        X_multiclass_scaled = self.multiclass_scaler.transform(X_multiclass)
        multiclass_predictions = self.multiclass_clf.predict_proba(X_multiclass_scaled)

        probas = CustomClassifier.transform_arr_to_proba(binary_predictions, multiclass_predictions)

        return probas











total_path = r"C:\Users\ayber\Desktop\Auswertung\Ausw_nur_cus.csv"
# total_path = r"C:\Users\ayber\Desktop\Auswertung\Ausw_both_methods.csv"

total_data_df = pd.read_csv(total_path)
classes = ["A", "C", "G", "T"]
# total_data_df = remove_outliers_iqr_by_class(total_data_df, bound=1, groupby_label='nucleotide')

x = total_data_df.iloc[:, [0, 1, 2, 3, 4]].values
y = total_data_df.iloc[:, -1].values



# y = np.where(y != 0, 1, y)

# c0_mask = np.where(y != 0)
# x = x[c0_mask]
# y = y[c0_mask]



from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# sc = StandardScaler()       #das scaling macht manchmal einen Unterschied !!!!
# # sc = MinMaxScaler(feature_range=(0, 1))
# x_train = sc.fit_transform(x_train)
# x_test = sc.transform(x_test)

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
oversampler = SMOTE(sampling_strategy={1: 1100, 2: 1100, 3: 1100})      #{1: 1100, 2: 1100, 3: 1100}
undersampler = RandomUnderSampler(sampling_strategy={1: 1000, 2: 1000, 3: 1000})       #{0: 1000, 1: 1000, 2: 1000, 3: 1000}

x_train, y_train = oversampler.fit_resample(x_train, y_train)
x_train, y_train = undersampler.fit_resample(x_train, y_train)

1
# y_train = np.where(y_train != 0, 1, y_train)
# y_test = np.where(y_test != 0, 1, y_test)



# c0_mask_train = np.where(y_train != 0)
# c0_mask_test = np.where(y_test != 0)
#
# x_train = x_train[c0_mask_train]
# y_train = y_train[c0_mask_train]
# x_test = x_test[c0_mask_test]
# y_test = y_test[c0_mask_test]
1



from sklearn.ensemble import RandomForestClassifier
rfc_binary = RandomForestClassifier(n_estimators=70, criterion="entropy")

from sklearn.ensemble import RandomForestClassifier
rfc_multi = RandomForestClassifier(n_estimators=40)


classifier = CustomClassifier(rfc_binary, rfc_multi)
classifier.fit(x_train, y_train)
# y_pred = classifier.predict(x_test)
y_proba = classifier.predict_proba(x_test)
y_pred = np.argmax(y_proba, axis=1)

classes = ["A", "nA"]
accuracy = plot_confusion_matr(y_test, y_pred, classes)






#
# from sklearn.ensemble import RandomForestClassifier
# rfc = RandomForestClassifier(n_estimators=70, criterion="entropy") #n_estimators=70, criterion="entropy"    #n_estimators=78, criterion="log_loss"
# rfc.fit(x_train, y_train)
# y_pred = rfc.predict(x_test)
#
# # from xgboost.sklearn import XGBClassifier
# # xgb = XGBClassifier(n_estimators=800, max_depth=12, learning_rate=0.05)
# # xgb.fit(x_train, y_train)
# # y_pred = xgb.predict(x_test)
#
#
# classes = ["A", "nA"]
# accuracy = plot_confusion_matr(y_test, y_pred, classes)
# from sklearn.metrics import f1_score
# f1_weighted = f1_score(y_test, y_pred, average='weighted')
# f1_macro = f1_score(y_test, y_pred, average='macro')
# f1_micro = f1_score(y_test, y_pred, average='micro')
#
# print("accuracy: ", accuracy)
# print("f1_weighted: ", f1_weighted)
# print("f1_macro: ", f1_macro)
# print("f1_micro: ", f1_micro)






exit()
from Ma_revamp.total.shap_calculation import calc_and_save_shap_values
exp_folder_name = "dev_data_new"
folder_path = rf"C:\Users\ayber\Desktop\shap_value_ordner\{exp_folder_name}"
filename = "custom_shap_values_test"
if not os.path.isdir(folder_path):
    os.mkdir(folder_path)
calc_and_save_shap_values(classifier, x, folder_path, filename=filename)   #sc.transform(x), x_train
























