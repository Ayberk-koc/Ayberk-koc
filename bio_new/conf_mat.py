import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec






ann_path = r"C:\Users\ayber\Desktop\conf_mat_arrs\bio\ann.npy"
cnn_path = r"C:\Users\ayber\Desktop\conf_mat_arrs\bio\cnn.npy"
rfc_path = r"C:\Users\ayber\Desktop\conf_mat_arrs\bio\rfc.npy"
xgb_path = r"C:\Users\ayber\Desktop\conf_mat_arrs\bio\xgb.npy"


ann_conf = np.load(ann_path)
cnn_conf = np.load(cnn_path)
xgb_conf = np.load(xgb_path)
rfc_conf = np.load(rfc_path)


classes = ["A3", "A4", "A5"]

fig = plt.figure(figsize=(10, 10))
gs = gridspec.GridSpec(2, 2,  width_ratios=[0.8, 1])

ax1 = fig.add_subplot(gs[0, 0])
sns.heatmap(ann_conf, annot=True, cmap='Blues', fmt="0.0%", xticklabels=classes, yticklabels=classes, annot_kws={"fontsize": 20}, cbar=False)
ax1.set_title("ANN")

ax2 = fig.add_subplot(gs[0, 1])
sns.heatmap(cnn_conf, annot=True, cmap='Blues', fmt="0.0%", xticklabels=classes, yticklabels=classes, annot_kws={"fontsize": 20}, cbar=True)
ax2.set_title("CNN")

ax3 = fig.add_subplot(gs[1, 0])
sns.heatmap(xgb_conf, annot=True, cmap='Blues', fmt="0.0%", xticklabels=classes, yticklabels=classes, annot_kws={"fontsize": 20}, cbar=False)
ax3.set_title("XGB")

ax4 = fig.add_subplot(gs[1, 1])
sns.heatmap(rfc_conf, annot=True, cmap='Blues', fmt="0.0%", xticklabels=classes, yticklabels=classes, annot_kws={"fontsize": 20}, cbar=True)
ax4.set_title("RFC")

# fig.subplots_adjust(hspace=0.3, wspace=0.3)
# fig.text(0.5, 0.95, 'Figure Title', ha='center', fontsize=16)
# fig.text(0.5, 0.0, 'X-axis Label', ha='center', fontsize=12)
# plt.tight_layout()  #hier noch immer 2-3 mal den hspace und vspace manuell einstellen!
plt.show()





exit()
plt.figure(figsize=(6, 6))
sns.heatmap(rfc_conf, annot=True, cmap='Blues', fmt="0.0%",
            xticklabels=classes, yticklabels=classes, annot_kws={"fontsize": 20},
            cbar=False
            )
# plt.xlabel('Predicted label', fontsize=14)
# plt.ylabel('True label', fontsize=14)
plt.title("RFC")
plt.tick_params(axis='both', which='both', labelsize=12)
plt.show()






























