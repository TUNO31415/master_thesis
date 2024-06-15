import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from utils import split_train_test

def compare_summary_and_retro(summary_folder_path, retro_csv_path):

    df_retro = pd.read_csv(retro_csv_path)

    retro_labels = {
        "MD" : [],
        "CI" : [],
        "FI" : [],
        "IC" : [],
        "P" : []
    }

    sum_labels = {
        "MD" : [],
        "CI" : [],
        "FI" : [],
        "IC" : [],
        "P" : []
    }

    file_list = []

    for sum_csv in os.listdir(summary_folder_path):
        if not sum_csv.endswith("csv"):
            continue
        
        file_name = sum_csv.split(".")[0]
        batch_num = "_".join(file_name.split("_")[6:8])
        self_pid = "_".join(file_name.split("_")[4:6])
        other_pid = "_".join(file_name.split("_")[10:12])

        if self_pid == other_pid:
            other_pid = "_".join(file_name.split("_")[8:10])

        dimensions = ["MD", "CI", "FI", "IC", "P"]
        df_summary = pd.read_csv(summary_folder_path + sum_csv)

        retro_row = df_retro[(df_retro["BatchNum"] == batch_num) & (df_retro["selfPID"] == self_pid) & (df_retro["otherPID"] == other_pid)]
        file_list.append(sum_csv)
        for d in dimensions:
            retro_val_tmp = retro_row.iloc[0][d]
            sum_val_tmp = df_summary.iloc[0][d]
            retro_labels[d].append(retro_val_tmp)
            sum_labels[d].append(sum_val_tmp)
        

    results = []
    output_folder = "/Users/taichi/Desktop/master_thesis/manual_inspection/"
    for d in dimensions:
        dim_retro = np.array(retro_labels[d])
        dim_sum = np.array(sum_labels[d])
        differences = np.absolute(dim_retro - dim_sum)
        max_id = np.argmax(differences)
        min_id = np.argmin(differences)

        results.append({
            "dimension" : d,
            "max_instance" : file_list[max_id],
            "max_summary_predicted" : dim_sum[max_id],
            "max_petrospective_correct" : dim_retro[max_id],
            "min_instance" : file_list[min_id],
            "min_summary_predicted" : dim_sum[min_id],
            "min_petrospective_correct" : dim_retro[min_id]
        })

        if not os.path.exists(output_folder + f"summary/{d}/"):
            os.mkdir(output_folder + f"summary/{d}/")
        
        target_name_max = file_list[max_id][len("score_only_"):]
        target_name_min = file_list[min_id][len("score_only_"):]

        os.system(f"cp /Users/taichi/Desktop/master_thesis/RealTimeSIS_summary_label/{target_name_max} /Users/taichi/Desktop/master_thesis/manual_inspection/summary/{d}/max_diff_{target_name_max}")
        os.system(f"cp /Users/taichi/Desktop/master_thesis/RealTimeSIS_summary_label/{target_name_min} /Users/taichi/Desktop/master_thesis/manual_inspection/summary/{d}/min_diff_{target_name_min}")

    
    df = pd.DataFrame(results)
    df.to_csv(output_folder + "retro_sum_max_min_per_dim.csv")

# Dimension_name = MD, CI, FI, IC, P
def data_loader(dimension_name, real_time_sis_folder_path, retrospective_sis_file_path="/Users/taichi/Desktop/master_thesis/retrospective_sis.csv"):
    retro_csv_df = pd.read_csv(retrospective_sis_file_path)

    X = []
    Y = []
    files = []

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
            files.append(file_name[len(real_time_sis_folder_path):])
    
    return X, Y, files

def smart_peak_end_rule(X_train, Y_train, X_test, Y_test):
    new_X_train = np.array([[max(x), x[-1]] for x in X_train])
    regressor = LinearRegression().fit(new_X_train, np.array(Y_train))
    # print(f"peak weight : {regressor.coef_[0]}, end weight : {regressor.coef_[1]}, intercept : {regressor.intercept_}")
    new_X_test = np.array([[max(x), x[-1]] for x in X_test])
    Y_pred = regressor.predict(new_X_test)

    return Y_pred

def main():
    repeat = 10
    n = 10
    dimensions = ["MD", "CI", "FI", "IC", "P"]
    output_folder = "/Users/taichi/Desktop/master_thesis/manual_inspection/"
    rt_sis_folder = "/Users/taichi/Desktop/master_thesis/RealTimeSIS_with_context_v2/score_only/"
    retro_sis_file = "/Users/taichi/Desktop/master_thesis/retrospective_sis.csv"

    results = []

    for d in dimensions:
        max_diff_instance = ""
        min_diff_instance = ""
        max_tmp = 0
        min_tmp = 10000
        max_y_pred = 0
        max_y_correct = 0
        min_y_pred = 0
        min_y_correct = 0

    
        X, Y, files = data_loader(d, rt_sis_folder, retrospective_sis_file_path=retro_sis_file)
        
        for re in range(repeat):
            kf =KFold(n_splits=n, shuffle=True)
            for fold, (train_ids, test_ids) in enumerate(kf.split(X, Y)):
                X_train, y_train, X_test, y_test = split_train_test(X, Y, train_ids, test_ids)
                Y_pred = smart_peak_end_rule(X_train, y_train, X_test, y_test)

                diff = np.absolute(Y_pred - y_test)

                if max_tmp < np.max(diff):
                    max_id = np.argmax(diff)
                    max_y_pred = Y_pred[max_id]
                    max_y_correct = y_test[max_id]
                    max_diff_instance = files[test_ids[max_id]]

                if min_tmp > np.min(diff):
                    min_id = np.argmin(diff)
                    min_y_pred = Y_pred[min_id]
                    min_y_correct = y_test[min_id]
                    min_diff_instance = files[test_ids[min_id]]

        results.append({
            "dimension" : d,
            "max_diff_instance" : max_diff_instance,
            "max_y_pred" : max_y_pred,
            "max_y_correct" : max_y_correct,
            "min_diff_instance" : min_diff_instance,
            "min_y_pred" : min_y_pred,
            "min_y_correct" : min_y_correct
        })

        if not os.path.exists(output_folder + f"estimated_rt/{d}/"):
            os.mkdir(output_folder + f"estimated_rt/{d}/")
        
        target_name_max = max_diff_instance[len("score_only_"):]
        target_name_min = min_diff_instance[len("score_only_"):]

        os.system(f"cp /Users/taichi/Desktop/master_thesis/RealTimeSIS_with_context_v2/{target_name_max} /Users/taichi/Desktop/master_thesis/manual_inspection/estimated_rt/{d}/max_diff_{target_name_max}")
        os.system(f"cp /Users/taichi/Desktop/master_thesis/RealTimeSIS_with_context_v2/{target_name_min} /Users/taichi/Desktop/master_thesis/manual_inspection/estimated_rt/{d}/min_diff_{target_name_min}")
        

    df = pd.DataFrame(results)
    df.to_csv(output_folder + "rt_max_min_diff_per_dim.csv")

if __name__ == "__main__":
    main()
    compare_summary_and_retro("/Users/taichi/Desktop/master_thesis/RealTimeSIS_summary_label/score_only/", "/Users/taichi/Desktop/master_thesis/retrospective_sis.csv")
    