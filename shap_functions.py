import shap
import numpy as np
import tensorflow as tf
import copy


def get_expls_from_npz(filepath, featurenames):
    """
    filepath:          path to .npz file where arrays are saved
    featurenames:      list. name of features used for model (has to be in correct order regarding the model input)

    return:            dict of explanations (for every class one explanation), explanation for total data
    """

    file = np.load(filepath)

    arrs = {}

    for arr in file.files:
        arrs[arr] = file[arr]

    total_expl = shap.Explanation(arrs[file.files[0]], arrs[file.files[1]], data=arrs[file.files[2]],
                                  feature_names=featurenames)

    explanations = {}
    if len(arrs[file.files[0]].shape) < 3:
        expl_0 = copy.copy(total_expl)

        expl_1 = copy.copy(total_expl)
        expl_1.values = -1 * expl_1.values
        expl_1.base_values = 1 - expl_1.values

        values_stacked_arr = np.empty((*total_expl.values.shape, 2))
        values_stacked_arr[:, :, 0] = total_expl.values
        values_stacked_arr[:, :, 1] = total_expl.values * -1

        total_expl.base_values = total_expl.base_values.ravel()
        base_stacked_arr = np.empty((*total_expl.base_values.shape, 2))
        base_stacked_arr[:, 0] = total_expl.base_values
        base_stacked_arr[:, 1] = 1 - total_expl.base_values

        total_expl.values = values_stacked_arr
        total_expl.base_values = base_stacked_arr

        explanations["expl_0"] = expl_0
        explanations["expl_1"] = expl_1

    else:
        for i in range(arrs[file.files[0]].shape[2]):
            explanations[f"expl_{i}"] = shap.Explanation(arrs[file.files[0]][:, :, i], arrs[file.files[1]][:, i],
                                                         data=arrs[file.files[2]], feature_names=featurenames)

    return explanations, total_expl


def calc_and_save_shap_values(model, x, folder_path, filename="shap_val"):
    """
    Parameters
    ----------
    model           fitted model, that you want to calculate shap values with
    x               data from which shap values are calculated. (Has to be 2D input)
    folder_path     folder where you want to store the resulting ".npz"-file
    filename        name of file (don't use ending! E.g only give "name" instead of for example "name.csv")

    Returns
    -------

    """
    if isinstance(model, tf.keras.Model):
        pred_func = model.predict
        for layer in model.layers:
            if type(layer).__name__ == 'Conv1D' or type(layer).__name__ == 'Conv2D':
                def f(y):
                    y = y.reshape(y.shape[0], *model.input_shape[1:])
                    return model.predict(y)
                pred_func = f
                break
    else:
        pred_func = model.predict_proba

    explainer = shap.explainers.Exact(pred_func, x)
    shap_values = explainer(x)

    file_path = folder_path+fr"\{filename}.npz"
    np.savez(file_path, values=shap_values.values, base_values=shap_values.base_values, data=shap_values.data)







# example:
#### get data
# data = pd.read_csv("features.csv")
#### read out features and labels used for fitting
# x = data.iloc[:, [0, 1, 2, 4, 5, 6, 7, 9]].values
# y = data.iloc[:, -1].values
#### build model
# sc = StandardScaler()
# x = sc.fit_transform(x)
# rfc_model = RandomForestClassifier()
# rfc_model.fit(x, y)
#### set featurenames and classnames for plots later on
# featurenames = ["dwell_I", "mean_I", "height_I", "nr_lvl_I", "dwell_T", "mean_T", "height_T", "nr_lvl_T"]
# classnames = ["1kbp dsDNA", "80nt ssDNA", "polylysine"]
#### call function to calculate shap values
# calc_and_save_shap_values(model, x, folder_path, file_name)
#### read out values and return explanations (explanations store shap values)
# explanaitions, total_explanation = get_expls_from_npz(filepath, featurenames):
#### get explanations for each class and list of explainers (used to do easier plotting)
# r_expl_0 = explanaitions["expl_0"]
# r_expl_1 = explanaitions["expl_1"]
# r_expl_2 = explanaitions["expl_2"]
# r_shap_value_in_list = [total_explanation.values[:, :, i] for i in range(3)]
#### example to plot bar plot
# shap.initjs()
# shap.summary_plot(r_shap_value_in_list, total_explanation.data, plot_type="bar", feature_names=featurenames, class_names=classnames, show=False)
# plt.show()
#### example to plot beeswarm plot:
# shap.plots.beeswarm(r_expl_0, show=False)
# plt.show()
#### there are many more plot functions


