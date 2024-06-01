import pandas as pd
import os
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

def split_train_test(X, y, train_ids, test_ids):
    y_np = np.array(y)
    X_train = []
    y_train = y_np[train_ids].tolist()
    for i in train_ids:
        X_train.append(X[i])
    
    X_test = []
    y_test = y_np[test_ids].tolist()
    for i in test_ids:
        X_test.append(X[i])

    return X_train, y_train, X_test, y_test

# Dimension_name = MD, CI, FI, IC, P
def data_loader(dimension_name, real_time_sis_folder_path, retrospective_sis_file_path="/Users/taichi/Desktop/master_thesis/retrospective_sis.csv"):
    
    retro_csv_df = pd.read_csv(retrospective_sis_file_path)

    X = []
    Y = []

    for index, row in retro_csv_df.iterrows():
        batch_num = row["BatchNum"]
        selfPID = row["selfPID"]
        otherPID = row["otherPID"]

        target_file_name1 = f"score_only_rt_SIS_{selfPID}_{batch_num}_{selfPID}_{otherPID}.csv"
        target_file_name2 = f"score_only_rt_SIS_{selfPID}_{batch_num}_{otherPID}_{selfPID}.csv"
        file_name = ""
        if os.path.exists(real_time_sis_folder_path + target_file_name1):
            file_name = real_time_sis_folder_path + target_file_name1
        elif os.path.exists(real_time_sis_folder_path + target_file_name2):
            file_name = real_time_sis_folder_path + target_file_name2
        else:
            continue
        
        real_csv_df = pd.read_csv(file_name)
        try:
            x = [float(a) for a in real_csv_df[dimension_name].tolist()]
        except:
            continue
        else:
            X.append(x)
            Y.append(float(row[dimension_name]))
    
    return X, Y

def evaluation_metrics(Y_true, Y_pred):
    return [r2_score(Y_true, Y_pred), mean_squared_error(Y_true, Y_pred)]

def real_time_labels_distribution(output_path, csv_file_path = "/Users/taichi/Desktop/master_thesis/RealTimeSIS_v3_score_only/"):

    all_md = []
    all_ci = []
    all_fi = []
    all_ic = []
    all_p = []

    for file in os.listdir(csv_file_path):
        if not file.endswith('.csv'):
            continue
        df = pd.read_csv(csv_file_path + file)
        df.reset_index(inplace=True)

        if not "MD" in df.columns:
            continue

        all_md.append(df["MD"].tolist())
        all_ci.append(df["CI"].tolist())
        all_fi.append(df["FI"].tolist())
        all_ic.append(df["IC"].tolist())
        all_p.append(df["P"].tolist())

    all = {
        "MD" : all_md,
        "CI" : all_ci,
        "FI" : all_fi,
        "IC" : all_ic,
        "P" : all_p
    }

    bin = [1,2,3,4,5]

    for key, value in all.items():
        value = np.concatenate(value).tolist()
        likert_count = [value.count(cat) for cat in bin]
        plt.bar(bin, likert_count, color='skyblue')
        plt.xlabel(f"Likert scale values")
        plt.ylabel('Frequency')
        plt.title(f"Hisotogram of estimated real-time SIS evaluation of {key} ")
        plt.savefig(output_path + f"{key}_estimated_real-time_SIS_hist_percat.png")
        plt.close()

def real_time_labels_distribution_new(output_path, rt_folder = "/Users/taichi/Desktop/master_thesis/RealTimeSIS_v3_score_only/"):
    dimensions = ["MD", "CI", "FI", "IC", "P"]

    all_md = []
    all_ci = []
    all_fi = []
    all_ic = []
    all_p = []

    for file in os.listdir(rt_folder):
        if not file.endswith("csv"):
            continue

        df = pd.read_csv(rt_folder + file)
        df.reset_index(inplace=True)

        if not "MD" in df.columns:
            continue

        all_md.append(df["MD"].tolist())
        all_ci.append(df["CI"].tolist())
        all_fi.append(df["FI"].tolist())
        all_ic.append(df["IC"].tolist())
        all_p.append(df["P"].tolist())

    all = {
        "MD" : all_md,
        "CI" : all_ci,
        "FI" : all_fi,
        "IC" : all_ic,
        "P" : all_p
    }

    bin = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    for key, value in all.items():
        value = np.concatenate(value).tolist()
        likert_count = [value.count(cat) for cat in bin]
        plt.bar(bin, likert_count, color='skyblue', edgecolor='black', width=0.5)  # bins determine the number of bins in the histogram
        plt.title(f'Histogram of real-time SIS evaluation of {key}')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.xticks(bin)
        plt.savefig(output_path + f"rt_SIS_{key}_hist.png")
        plt.close()
        print(f"----- SAVED TO {output_path}rt_SIS_{key}_hist.png -----")
    
def retro_labels_distribution(output_path, csv_file_path = "/Users/taichi/Desktop/master_thesis/retrospective_sis.csv"):
    dimensions = ["MD", "CI", "FI", "IC", "P"]
    df = pd.read_csv(csv_file_path)
    bin = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    for d in dimensions:
        lis = df[d].tolist()
        likert_count = [lis.count(cat) for cat in bin]
        plt.bar(bin, likert_count, color='skyblue', edgecolor='black', width=0.5)  # bins determine the number of bins in the histogram
        plt.title(f'Histogram of retrospective SIS evaluation of {d}')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.xticks(bin)
        plt.savefig(output_path + f"retro_SIS_{d}_hist.png")
        plt.close()
        print(f"----- SAVED TO {output_path}retro_SIS_{d}_hist.png -----")
        
    
def read_token(file_path):
    with open(file_path, 'r') as file:
        token = file.read().strip()
    return token