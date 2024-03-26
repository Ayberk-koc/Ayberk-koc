import shap
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec







def plot_shap(shap_dict: dict, **kwargs):

    xgb_vals = shap_dict["xgb"]
    rfc_vals = shap_dict["rfc"]
    ann_vals = shap_dict["ann"]
    cnn_vals = shap_dict["cnn"]


    fig = plt.figure(figsize=(14, 14))
    gs = gridspec.GridSpec(2, 2)

    elems = sorted(zip(ann_vals, featurenames), key=lambda x: x[0],
                   reverse=False)  # bei plotten wir reversen aus irgendeinem grund
    values_sort = [elem[0] for elem in elems]
    features_sort = [elem[1] for elem in elems]

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.barh(np.arange(len(values_sort)), values_sort, height=0.8)  # plot bars
    ax1.set_yticks(np.arange(len(values_sort)))  # set ticks for labels
    ax1.set_yticklabels(features_sort)  # set labels on ticks
    ax1.tick_params(axis='x', labelsize=12)
    ax1.tick_params(axis='y', labelsize=18)
    ax1.set_xticks([0, 0.05, 0.1, 0.15, 0.2, 0.25])
    # ax1.xaxis.set_ticks_position('none')  # remove ticks (keine schwarzen strichlein)
    ax1.yaxis.set_ticks_position('none')
    ax1.set_ylim(bottom=-1)  # für padding nach unten
    ax1.grid(False)
    ax1.set_title("ANN", fontsize=15)




    elems = sorted(zip(cnn_vals, featurenames), key=lambda x: x[0],
                   reverse=False)  # bei plotten wir reversen aus irgendeinem grund
    values_sort = [elem[0] for elem in elems]
    features_sort = [elem[1] for elem in elems]

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.barh(np.arange(len(values_sort)), values_sort, height=0.8)  # plot bars
    ax2.set_yticks(np.arange(len(values_sort)))  # set ticks for labels
    ax2.set_yticklabels(features_sort)  # set labels on ticks
    ax2.tick_params(axis='x', labelsize=12)
    ax2.tick_params(axis='y', labelsize=18)
    ax2.set_xticks([0, 0.05, 0.1, 0.15, 0.2])
    # ax2.xaxis.set_ticks_position('none')  # remove ticks (keine schwarzen strichlein)
    ax2.yaxis.set_ticks_position('none')
    ax2.set_ylim(bottom=-1)  # für padding nach unten
    ax2.grid(False)
    ax2.set_title("CNN", fontsize=15)
    # plt.xlabel('permutation importance')
    # plt.show()




    elems = sorted(zip(xgb_vals, featurenames), key=lambda x: x[0],
                   reverse=False)  # bei plotten wir reversen aus irgendeinem grund
    values_sort = [elem[0] for elem in elems]
    features_sort = [elem[1] for elem in elems]

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.barh(np.arange(len(values_sort)), values_sort, height=0.8)  # plot bars
    ax3.set_yticks(np.arange(len(values_sort)))  # set ticks for labels
    ax3.set_yticklabels(features_sort)  # set labels on ticks
    ax3.tick_params(axis='x', labelsize=12)
    ax3.tick_params(axis='y', labelsize=18)
    ax3.set_xticks([0, 0.05, 0.1, 0.15, 0.2, 0.25])
    # ax3.xaxis.set_ticks_position('none')  # remove ticks (keine schwarzen strichlein)
    ax3.yaxis.set_ticks_position('none')
    ax3.set_ylim(bottom=-1)  # für padding nach unten
    ax3.grid(False)
    ax3.set_title("XGB", fontsize=15)
    # plt.xlabel('permutation importance')
    # plt.show()




    elems = sorted(zip(rfc_vals, featurenames), key=lambda x: x[0],
                   reverse=False)  # bei plotten wir reversen aus irgendeinem grund
    values_sort = [elem[0] for elem in elems]
    features_sort = [elem[1] for elem in elems]

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.barh(np.arange(len(values_sort)), values_sort, height=0.8)  # plot bars
    ax4.set_yticks(np.arange(len(values_sort)))  # set ticks for labels
    ax4.set_yticklabels(features_sort)  # set labels on ticks
    ax4.tick_params(axis='x', labelsize=12)
    ax4.tick_params(axis='y', labelsize=18)
    ax4.set_xticks([0, 0.05, 0.1, 0.15, 0.2])
    #ax4.xaxis.set_ticks_position('none')  # remove ticks (keine schwarzen strichlein)
    ax4.yaxis.set_ticks_position('none')
    ax4.set_ylim(bottom=-1)  # für padding nach unten
    ax4.grid(False)
    ax4.set_title("RFC", fontsize=15)
    # plt.xlabel('permutation importance')


    # fig.subplots_adjust(hspace=0.4)
    # fig.suptitle("ChipA: Cusum", fontsize=16)
    # plt.savefig(r"C:\Users\ayber\Desktop\Ma_Latex\Ma\imgs\ChipB_ft.png")    #, bbox_inches="tight"
    # fig.suptitle('mean |SHAP|', x=0.5, y=0.05, ha='center')   #mache das mit inkscape
    plt.show()




def get_expls_from_npz(file, featurenames):
    """
    file:              .npz object where arrays are saved (use:  npz_object = np.load("filename.npz")
    featurenames:      name of features used for model

    return:            dict of explanations, explainer for total data
    """

    arrs = {}

    for arr in file.files:
        arrs[arr] = file[arr]


    total_expl = shap.Explanation(arrs[file.files[0]], arrs[file.files[1]], data=arrs[file.files[2]], feature_names=featurenames)

    if len(arrs[file.files[0]].shape) < 3:  # 3D array heißt, mehrere klassen (nicht nur binary)
        return total_expl

    else:
        explanations = {}
        for i in range(arrs[file.files[0]].shape[2]):
            explanations[f"expl_{i}"] = shap.Explanation(arrs[file.files[0]][:, :, i], arrs[file.files[1]][:, i], data=arrs[file.files[2]], feature_names=featurenames)

        return explanations, total_expl


ann = r"C:\Users\ayber\Desktop\shap_value_ordner\bio_new\ann_shap_values_gut_K.npz"
cnn = r"C:\Users\ayber\Desktop\shap_value_ordner\bio_new\cnn_shap_values_gut_K.npz"
rfc = r"C:\Users\ayber\Desktop\shap_value_ordner\bio_new\rfc_shap_values_gut_K.npz"
xgb = r"C:\Users\ayber\Desktop\shap_value_ordner\bio_new\xgb_shap_values_gut_K.npz"



classnames = ["A3", "A4", "A5"]
featurenames = ["dwell", "mean", "height", "nr_lvls", "std"]
nr_feat = len(featurenames)



ann_file = np.load(ann)
ann_expl, ann_total = get_expls_from_npz(ann_file, featurenames=featurenames)
ann_shap_value_in_list = [ann_total.values[:, :, i] for i in range(len(classnames))]

cnn_file = np.load(cnn)
cnn_expl, cnn_total = get_expls_from_npz(cnn_file, featurenames=featurenames)
cnn_shap_value_in_list = [cnn_total.values[:, :, i] for i in range(len(classnames))]

rfc_file = np.load(rfc)
rfc_expl, rfc_total = get_expls_from_npz(rfc_file, featurenames=featurenames)
rfc_shap_value_in_list = [rfc_total.values[:, :, i] for i in range(len(classnames))]

xgb_file = np.load(xgb)
xgb_expl, xgb_total = get_expls_from_npz(xgb_file, featurenames=featurenames)
xgb_shap_value_in_list = [xgb_total.values[:, :, i] for i in range(len(classnames))]


xgb_vals = []
for i in range(len(classnames)):
    #if i != 1: continue
    arr = xgb_shap_value_in_list[i]
    mean_abs_values = np.mean(np.abs(arr), axis=0)
    xgb_vals.append(mean_abs_values)

rfc_vals = []
for i in range(len(classnames)):
    #if i != 1: continue
    arr = rfc_shap_value_in_list[i]
    mean_abs_values = np.mean(np.abs(arr), axis=0)
    rfc_vals.append(mean_abs_values)

ann_vals = []
for i in range(len(classnames)):
    #if i != 1: continue
    arr = ann_shap_value_in_list[i]
    mean_abs_values = np.mean(np.abs(arr), axis=0)
    ann_vals.append(mean_abs_values)

cnn_vals = []
for i in range(len(classnames)):
    #if i != 1: continue
    arr = cnn_shap_value_in_list[i]
    mean_abs_values = np.mean(np.abs(arr), axis=0)
    cnn_vals.append(mean_abs_values)


fig = plt.figure(figsize=(14, 14))
gs = gridspec.GridSpec(2, 2)


ax1 = fig.add_subplot(gs[0, 0])
data1 = ann_vals[0]
data2 = ann_vals[1]
data3 = ann_vals[2]
data_tot = data1 + data2 + data3
order = sorted(range(len(data_tot)), key=lambda k: data_tot[k], reverse=False)
data1 = np.array([data1[i] for i in order])
data2 = np.array([data2[i] for i in order])
data3 = np.array([data3[i] for i in order])
ax1.barh(range(len(data1)), data1)
ax1.barh(range(len(data2)), data2, left=data1)
ax1.barh(range(len(data3)), data3, left=data1+data2)
ax1.tick_params(axis='x', labelsize=12)
ax1.tick_params(axis='y', labelsize=18)
ax1.set_yticklabels([" "] + [featurenames[i] for i in order])
ax1.yaxis.set_ticks_position('none')
ax1.set_ylim(bottom=-1)
ax1.grid(False)
ax1.set_title("ANN", fontsize=15)



ax2 = fig.add_subplot(gs[0, 1])
data1 = cnn_vals[0]
data2 = cnn_vals[1]
data3 = cnn_vals[2]
data_tot = data1 + data2 + data3
order = sorted(range(len(data_tot)), key=lambda k: data_tot[k], reverse=False)
data1 = np.array([data1[i] for i in order])
data2 = np.array([data2[i] for i in order])
data3 = np.array([data3[i] for i in order])
ax2.barh(range(len(data1)), data1, label='A3')
ax2.barh(range(len(data2)), data2, left=data1, label='A4')
ax2.barh(range(len(data3)), data3, left=data1+data2, label='A5')
ax2.tick_params(axis='x', labelsize=12)
ax2.tick_params(axis='y', labelsize=18)
ax2.set_yticklabels([" "] + [featurenames[i] for i in order])
ax2.yaxis.set_ticks_position('none')
ax2.set_ylim(bottom=-1)
ax2.grid(False)
ax2.legend(prop={'size': 16})
ax2.set_title("CNN", fontsize=15)



ax3 = fig.add_subplot(gs[1, 0])
data1 = xgb_vals[0]
data2 = xgb_vals[1]
data3 = xgb_vals[2]
data_tot = data1 + data2 + data3
order = sorted(range(len(data_tot)), key=lambda k: data_tot[k], reverse=False)
data1 = np.array([data1[i] for i in order])
data2 = np.array([data2[i] for i in order])
data3 = np.array([data3[i] for i in order])
ax3.barh(range(len(data1)), data1)
ax3.barh(range(len(data2)), data2, left=data1)
ax3.barh(range(len(data3)), data3, left=data1+data2)
ax3.tick_params(axis='x', labelsize=12)
ax3.tick_params(axis='y', labelsize=18)
ax3.set_yticklabels([" "] + [featurenames[i] for i in order])
ax3.yaxis.set_ticks_position('none')
ax3.set_ylim(bottom=-1)
ax3.grid(False)
ax3.set_title("RFC", fontsize=15)


ax4 = fig.add_subplot(gs[1, 1])
data1 = rfc_vals[0]
data2 = rfc_vals[1]
data3 = rfc_vals[2]
data_tot = data1 + data2 + data3
order = sorted(range(len(data_tot)), key=lambda k: data_tot[k], reverse=False)
data1 = np.array([data1[i] for i in order])
data2 = np.array([data2[i] for i in order])
data3 = np.array([data3[i] for i in order])
ax4.barh(range(len(data1)), data1)
ax4.barh(range(len(data2)), data2, left=data1)
ax4.barh(range(len(data3)), data3, left=data1+data2)
ax4.tick_params(axis='x', labelsize=12)
ax4.tick_params(axis='y', labelsize=18)
ax4.set_yticklabels([" "] + [featurenames[i] for i in order])
ax4.yaxis.set_ticks_position('none')
ax4.set_ylim(bottom=-1)
ax4.grid(False)
ax4.set_title("XGB", fontsize=15)




plt.show()




exit()











exit()

fig = plt.figure(figsize=(14, 14))
gs = gridspec.GridSpec(2, 2)

elems = sorted(zip(ann_vals, featurenames), key=lambda x: x[0],
               reverse=False)  # bei plotten wir reversen aus irgendeinem grund
values_sort = [elem[0] for elem in elems]
features_sort = [elem[1] for elem in elems]

ax1 = fig.add_subplot(gs[0, 0])
ax1.barh(np.arange(len(values_sort)), values_sort, height=0.8)  # plot bars
ax1.set_yticks(np.arange(len(values_sort)))  # set ticks for labels
ax1.set_yticklabels(features_sort)  # set labels on ticks
ax1.tick_params(axis='x', labelsize=12)
ax1.tick_params(axis='y', labelsize=18)
ax1.set_xticks([0, 0.05, 0.1, 0.15, 0.2, 0.25])
# ax1.xaxis.set_ticks_position('none')  # remove ticks (keine schwarzen strichlein)
ax1.yaxis.set_ticks_position('none')
ax1.set_ylim(bottom=-1)  # für padding nach unten
ax1.grid(False)
ax1.set_title("ANN", fontsize=15)









exit()

plot_shap(shap_vals)



exit()
elems = sorted(zip(mean_abs_values, featurenames), key=lambda x: x[0],
               reverse=False)  # bei plotten wir reversen aus irgendeinem grund
values_sort = [elem[0] for elem in elems]
features_sort = [elem[1] for elem in elems]

fig, ax = plt.subplots(figsize=(8, 3))
ax.barh(np.arange(len(values_sort)), values_sort, height=0.8)  # plot bars
ax.set_yticks(np.arange(len(values_sort)))  # set ticks for labels
ax.set_yticklabels(features_sort)  # set labels on ticks
ax.xaxis.set_ticks_position('none')  # remove ticks (keine schwarzen strichlein)
ax.yaxis.set_ticks_position('none')
ax.set_ylim(bottom=-1)  # für padding nach unten
ax.grid(False)
plt.xlabel('permutation importance')
plt.show()


















