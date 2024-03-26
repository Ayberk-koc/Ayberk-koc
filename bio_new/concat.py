import pandas as pd

def concat(path_list, label_col=None, ):
    dfs = []
    for i, path in enumerate(path_list):
        df_temp = pd.read_csv(path)
        if label_col: df_temp[label_col] = i
        dfs.append(df_temp)
    final_df = pd.concat([*dfs], axis=0)
    return final_df





path1 = r"C:\Users\ayber\Desktop\pelt_test\Bio_R\features_bio_A3R.csv"
path2 = r"C:\Users\ayber\Desktop\pelt_test\Bio_R\features_bio_A4R.csv"
path3 = r"C:\Users\ayber\Desktop\pelt_test\Bio_R\features_bio_A5R.csv"
#path4 = r"C:\Users\ayber\Desktop\pelt_test\8ter_try_l2\DevB_T\features_DevB_T.csv"

paths = [path1, path2, path3,
         #path4,
         ]


final_df = concat(paths, label_col="label")
save_path = r"C:\Users\ayber\Desktop\pelt_test\Bio_R\bio_R.csv"
final_df.to_csv(save_path, index=False)




