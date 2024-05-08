import pandas as pd
import os
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error


real_time_sis_folder_path = "/Users/taichi/Desktop/master_thesis/RealTimeSIS_v3_score_only/"
retrospective_sis_file_path = "/Users/taichi/Desktop/master_thesis/retrospective_sis.csv"

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
def data_loader(dimension_name):
    
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