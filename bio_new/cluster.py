from sklearn.cluster import OPTICS, HDBSCAN
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.gridspec as gridspec
from sklearn.metrics import silhouette_score

def plot_fts(features: tuple, **kwargs):
    """
        args has to be given like this: (mean, height, dwell)
        kwargs are additional argument for the fig.scatter - method
        wenn ich "clus" mitgebe, dann werden die ellipsen erstellt!
    """
    mean = features[0]
    height = features[1]
    dwell = features[2]
    num_lvls = features[3]

    fig = plt.figure(figsize=(14, 14))
    gs = gridspec.GridSpec(2, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(mean, height, s=20, **kwargs)
    ax1.set_xlabel(f'mean [nA]', fontsize=14)
    ax1.set_ylabel(f'height [nA]', fontsize=14)
    ax1.tick_params(axis='x', labelsize=12)
    ax1.tick_params(axis='y', labelsize=12)
    ax1.grid(linewidth=0.5, alpha=0.5, color='gray')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(dwell, mean, s=20, **kwargs)
    ax2.set_xlabel(f'dwell [ms]', fontsize=12)
    ax2.set_ylabel(f'mean [nA]', fontsize=12)
    ax2.tick_params(axis='x', labelsize=12)
    ax2.tick_params(axis='y', labelsize=12)
    ax2.grid(linewidth=0.5, alpha=0.5, color='gray')

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.scatter(dwell, height, s=20, **kwargs)
    ax3.set_xlabel(f'dwell [ms]', fontsize=12)
    ax3.set_ylabel(f'height [nA]', fontsize=12)
    ax3.tick_params(axis='x', labelsize=12)
    ax3.tick_params(axis='y', labelsize=12)
    ax3.grid(linewidth=0.5, alpha=0.5, color='gray')



    ax4 = fig.add_subplot(gs[1, 1])
    ax4.scatter(range(len(reachability_distances)), reachability_distances, marker='.')
    ax4.axhline(y=0.2, color='red', linestyle='-.', alpha=0.5, label="nr clusters = 8", xmin=0, xmax=0)    #, xmin=0, xmax=0    das nur für A5K
    #ax4.text(0, 0.5, 'Above the line', ha='center')
    #ax4.axhline(y=0.15, color='grey', linestyle='-.', alpha=0.5, label="nr clusters = 3")
    ax4.set_xlabel('Data Points', fontsize=12)
    ax4.set_ylabel(f'Reachability Distance (ε), n={min_samp}', fontsize=12)
    ax4.tick_params(axis='x', labelsize=12)
    ax4.tick_params(axis='y', labelsize=12)
    ax4.legend(fontsize=15)
    ax4.grid(linewidth=0.5, alpha=0.5, color='gray')
    ## ax4.title('Reachability Distance n=50')

    # ax4 = fig.add_subplot(gs[1, 1])       #hier für num_lvls plot
    # ax4.scatter(num_lvls, height, s=20, **kwargs)
    # ax4.set_xlabel(f'num_lvls #')
    # ax4.set_ylabel(f'height [nA]')

    fig.subplots_adjust(hspace=0.4)
    # fig.suptitle("ChipA: Cusum", fontsize=16)
    # plt.savefig(r"C:\Users\ayber\Desktop\Ma_Latex\Ma\imgs\ChipB_ft.png")    #, bbox_inches="tight"
    plt.show()



def reachability_values(x, min_samples, eps=1, show=False):
    """
    x müssen dieselben daten sein die ich clustere!
    Auch das min_samples muss dasselbe wie beim clustern sein!
    """
    clust = OPTICS(cluster_method="dbscan", max_eps=eps, min_samples=min_samples)
    clust.fit(x)
    reachability_distances = clust.reachability_[clust.ordering_]
    # core_distances = clust.core_distances_[clust.ordering_]

    if show:
        plt.figure(figsize=(10, 6))
        plt.scatter(range(len(reachability_distances)), reachability_distances, marker='.')
        plt.axhline(y=0.3, color='red', linestyle='-.', alpha=0.5, label="nr clusters = 5")
        plt.axhline(y=0.15, color='grey', linestyle='-.', alpha=0.5, label="nr clusters = 3")
        plt.xlabel('Data Points')
        plt.ylabel('Reachability Distance')
        plt.title('Reachability Distance n=50')
        plt.legend()
        plt.grid(True)
        plt.show()

    return reachability_distances



PoreB = r"C:\Users\ayber\Desktop\pelt_test\PoreB_J\features_PoreB_10MHz.csv"
PoreJ = r"C:\Users\ayber\Desktop\pelt_test\PoreB_J\features_PoreJ_10MHz.csv"
ChipA = r"C:\Users\ayber\Desktop\pelt_test\Chips_new\features_ChipA_scaled.csv"
ChipB = r"C:\Users\ayber\Desktop\pelt_test\Chips_new\features_ChipB.csv"
DevB_A = r"C:\Users\ayber\Desktop\Auswertung\gut_aktuell(nurB)_mit_lvl_info\DevB_A.csv"



Bio_3K_120 = r"C:\Users\ayber\Desktop\test\bio_nutzen\A3_K_0-10kHz_120.csv"
Bio_3K_120_eng = r"C:\Users\ayber\Desktop\test\bio_nutzen\A3_K_0-10kHz_120_eng.csv"
Bio_3K_120_eng2 = r"C:\Users\ayber\Desktop\test\bio_nutzen\A3_K_0-10kHz_120_eng2.csv"
Bio_4K_120 = r"C:\Users\ayber\Desktop\test\bio_nutzen\A4_K_0-10kHz_120.csv"
Bio_5K_120 = r"C:\Users\ayber\Desktop\test\bio_nutzen\A5_K_0-10kHz_120.csv"
Bio_5K_120_eng = r"C:\Users\ayber\Desktop\test\bio_nutzen\A5_K_0-10kHz_120_eng.csv"
Bio_5K_120_eng2 = r"C:\Users\ayber\Desktop\test\bio_nutzen\A5_K_0-10kHz_120_eng2.csv"

data = Bio_5K_120_eng2
df = pd.read_csv(data)
# df = df[df['dwell_time'] <= 10e-3]      #um werte zu filtern

# df = df[df['mean'] <= 1.5e-9]       #um werte zu filtern
# df = df[df['height'] <= 1.5e-9]     #um werte zu filtern

mean = df["mean"] * 10**9                #in nA
height = df["height"] * 10**9            #in nA
dwell = df["dwell_time"] * 10**3         #in ms
num_lvls = df["num_lvls"]



# plot_fts((mean, height, dwell, num_lvls), cmap="plasma")





num_lvls = num_lvls.map(lambda x: 2 if x > 1 else 0)    #hier um die mit hohen lvls von einander zu trennen
scaler = StandardScaler()
x = np.stack((mean, height, num_lvls), axis=1)        #, num_lvls
x[:, :2] = scaler.fit_transform(x[:, :2])   #scale nicht die anzahl an lvl -> points innerhalb cluster müssen same num_lvls haben!

min_samp = 30  #15 für A4K & A3K, 30 für A5K
reachability_distances = reachability_values(x, min_samples=min_samp, eps=1, show=False)


# opt = OPTICS(min_samples=50, xi=0.05, min_cluster_size=0.05)          #hier lese noch etwas wie man die params genau einstellen sollt
eps_clus = 0.2
# clust = OPTICS(cluster_method="xi", max_eps=0.2, min_samples=min_samp, eps=eps_clus, xi=0.01, min_cluster_size=75)
# clust = OPTICS(cluster_method="dbscan", max_eps=0.42, min_samples=min_samp, eps=eps_clus)             #für A4K
# clust = HDBSCAN(min_samples=3, min_cluster_size=50, cluster_selection_epsilon=0.1, alpha=2)         #für A3K
clust = HDBSCAN(min_samples=5, min_cluster_size=75)                                                 #für A5K
clust.fit(x)
labels_init = clust.labels_

# for i in range(len(labels_init)):
#     if labels_init[i] == 0:
#         labels_init[i] = 6

#####für A4K zum farbe anpassen. Kann Farbe ändern indem ich label ändere!
for i in range(len(labels_init)):
    if labels_init[i] == 2:
        labels_init[i] = 1
    elif labels_init[i] == 1:
        labels_init[i] = 0
    elif labels_init[i] == 0:
        labels_init[i] = 2


# silhouette_avg = silhouette_score(x[labels_init > -1], labels_init[labels_init > -1])
# print(silhouette_avg)



# cmap = plt.get_cmap('viridis')
# label_to_change = 2
# new_color = 'grey'
# from matplotlib.colors import ListedColormap
# # Create a custom colormap with the modified color for the specific label
# colors = [cmap(i) for i in np.linspace(0, 1, cmap.N)]
# colors[label_to_change] = plt.cm.colors.to_rgba(new_color)
# custom_cmap = ListedColormap(colors)




plot_fts((mean, height, dwell, num_lvls), c=labels_init, alpha=1)   #cmap="plasma", custom_map



