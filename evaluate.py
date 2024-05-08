import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from scipy.stats import t

real_time_sis_folder_path = "/Users/taichi/Desktop/master_thesis/RealTimeSIS_v3_score_only/"
retrospective_sis_file_path = "/Users/taichi/Desktop/master_thesis/retrospective_sis.csv"

def peak_end_rule(X, Y):
    Y_pred = [(max(x) + x[-1])/2 for x in X]
    return evaluation_metrics(Y, Y_pred)

def peak_only(X, Y):
    Y_pred = [max(x) for x in X]
    return evaluation_metrics(Y, Y_pred)

def end_only(X, Y):
    Y_pred = [x[-1] for x in X] 
    return evaluation_metrics(Y, Y_pred)

def base_line(X, Y):
    Y_pred = [sum(x) / len(x) for x in X]
    return evaluation_metrics(Y, Y_pred)

def evaluation_metrics(Y_true, Y_pred):
    return [r2_score(Y_true, Y_pred), mean_squared_error(Y_true, Y_pred)]

def print_evaluation_result(model_name, dimension, eval_list):
    print(f"{model_name} {dimension} | R2 : {eval_list[0]} | MSE : {eval_list[1]}")

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

def output_results_all_dimensions(output_path):
    dimensions = ["MD", "CI", "FI", "IC", "P"]
    entries = []

    for d in dimensions:
        X, Y = data_loader(d)
        result_pe = peak_end_rule(X, Y)
        result_p = peak_only(X, Y)
        result_e = end_only(X, Y)
        result_base = base_line(X, Y)
        entry = {"dimension" : d, "Peak-End R^2" : result_pe[0], "Peak-End MSE" : result_pe[1], "Peak-Only R^2" : result_p[0], "Peak-Only MSE" : result_p[1], "End-Only R^2" : result_e[0], "End-Only MSE" : result_e[1], "Base R^2" : result_base[0], "Base MSE" : result_base[1]}
        entries.append(entry)
    
    df = pd.DataFrame(entries)
    df.to_csv(output_path)
    print(f"SAVED TO {output_path}")

# difference_mode = True : Returns all 10x10 performance metrics in a vector
# difference_mode = False : Returns mean and std of 10x10 performance metrics
def output_results_all_dimensions_kfold(n=10, repeat=10, difference_mode=False):
    dimensions = ["MD", "CI", "FI", "IC", "P"]
    entries_mse = []
    entries_r2 = []
    entries = []
    for d in dimensions:
        for re in range(repeat):
            X, Y = data_loader(d)
            kf = KFold(n_splits=n, shuffle=True)
            results_pe_r = []
            results_pe_m = []
            results_p_r = []
            results_p_m = []
            results_e_r = []
            results_e_m = []
            results_base_r = []
            results_base_m = []
            for fold, (train_index, test_index) in enumerate(kf.split(X, Y)):
                x_test = []
                y_np = np.array(Y)
                y_test = y_np[test_index].tolist()
                for i in test_index:
                    x_test.append(X[i])
                
                results_pe_r.append(peak_end_rule(x_test, y_test)[0])
                results_pe_m.append(peak_end_rule(x_test, y_test)[1])
                results_p_r.append(peak_only(x_test, y_test)[0])
                results_p_m.append(peak_only(x_test, y_test)[1])
                results_e_r.append(end_only(x_test, y_test)[0])
                results_e_m.append(end_only(x_test, y_test)[1])
                results_base_r.append(base_line(x_test, y_test)[0])
                results_base_m.append(base_line(x_test, y_test)[1])
                print(f"{d} : REPEAT {re+1} in {fold+1}th FOLD")

        if difference_mode:
            entry = {
                "Peak-End R^2" : results_pe_r,
                "Peak-Only R^2" : results_p_r,
                "End-Only R^2" : results_e_r,
                "Base R^2" : results_base_r,
                "Peak-End MSE" : results_pe_m,
                "Peak-Only MSE" : results_p_m,
                "End-Only MSE" : results_e_m,
                "Base MSE" : results_base_m,
            }
            entries.append(entry)
        else:
            entry_r2 = {"Dimension" : d, 
                    "Peak-End R^2 mean" : np.mean(results_pe_r), 
                    "Peak-End R^2 std" : np.std(results_pe_r), 
                    "Peak-Only R^2 mean" : np.mean(results_p_r), 
                    "Peak-Only R^2 std" : np.std(results_p_r), 
                    "End-Only R^2 mean" : np.mean(results_e_r),
                    "End-Only R^2 std" : np.std(results_e_r),
                    "Base R^2 mean" : np.mean(results_base_r),
                    "Base R^2 std" : np.std(results_base_r),
                    }

            entry_mse = {"Dimension" : d, 
                    "Peak-End MSE mean" : np.mean(results_pe_m),
                    "Peak-End MSE std" : np.std(results_pe_m),
                    "Peak-Only MSE mean" : np.mean(results_p_m),
                    "Peak-Only MSE std" : np.std(results_p_m),
                    "End-Only MSE mean" : np.mean(results_e_m),
                    "End-Only MSE std" : np.std(results_e_m),
                    "Base MSE mean" : np.mean(results_base_m),
                    "Base MSE std" : np.std(results_base_m)
                    }
            entries_r2.append(entry_r2)
            entries_mse.append(entry_mse)
    
    if difference_mode:
        df = pd.DataFrame(entries)
        return df
    else:
        df_r2 = pd.DataFrame(entries_r2)
        df_mse = pd.DataFrame(entries_mse)
        return df_r2, df_mse

def corrected_std(differences, n_train, n_test):
    """Corrects standard deviation using Nadeau and Bengio's approach.

    Parameters
    ----------
    differences : ndarray of shape (n_samples,)
        Vector containing the differences in the score metrics of two models.
    n_train : int
        Number of samples in the training set.
    n_test : int
        Number of samples in the testing set.

    Returns
    -------
    corrected_std : float
        Variance-corrected standard deviation of the set of differences.
    """
    # kr = k times r, r times repeated k-fold crossvalidation,
    # kr equals the number of times the model was evaluated
    kr = len(differences)
    corrected_var = np.var(differences, ddof=1) * (1 / kr + n_test / n_train)
    corrected_std = np.sqrt(corrected_var)
    return corrected_std

def compute_corrected_ttest(differences, df, n_train, n_test):
    """Computes right-tailed paired t-test with corrected variance.

    Parameters
    ----------
    differences : array-like of shape (n_samples,)
        Vector containing the differences in the score metrics of two models.
    df : int
        Degrees of freedom.
    n_train : int
        Number of samples in the training set.
    n_test : int
        Number of samples in the testing set.

    Returns
    -------
    t_stat : float
        Variance-corrected t-statistic.
    p_val : float
        Variance-corrected p-value.
    """
    mean = np.mean(differences)
    std = corrected_std(differences, n_train, n_test)
    t_stat = mean / std
    p_val = t.sf(np.abs(t_stat), df)  # right-tailed t-test
    return t_stat, p_val

def save_figs_tables(output_folder):
    df_r2, df_mse = output_results_all_dimensions_kfold()
    df_r2.to_csv(output_folder + f"result_r2.csv")
    df_mse.to_csv(output_folder + f"result_MSE.csv")
    x_axis = np.array(["peak-end", "peak-only", "end-only", "baseline"])

    for index, row in df_mse.iterrows():
        y_axis = np.array([row["Peak-End MSE mean"], row["Peak-Only MSE mean"], row["End-Only MSE mean"], row["Base MSE mean"]])
        err = np.array([row["Peak-End MSE std"], row["Peak-Only MSE std"], row["End-Only MSE std"], row["Base MSE std"]])
        err = err / 10
        plt.errorbar(x_axis, y_axis, err, linestyle='None', marker='^')
        plt.title(f"{row["Dimension"]}_MSE_result")
        plt.savefig(output_folder + f"{row["Dimension"]}_MSE_result.png")
        plt.close()

        
    for index, row in df_r2.iterrows():
        y_axis = np.array([row[f"Peak-End R^2 mean"], row["Peak-Only R^2 mean"], row["End-Only R^2 mean"], row["Base R^2 mean"]])
        err = np.array([row["Peak-End R^2 std"], row["Peak-Only R^2 std"], row["End-Only R^2 std"], row["Base R^2 std"]])
        err = err / 10
        plt.errorbar(x_axis, y_axis, err, linestyle='None', marker='^')
        plt.title(f"{row["Dimension"]}_R^2_result")
        plt.savefig(output_folder + f"{row["Dimension"]}_R^2_result.png")
        plt.close()

# def save_corrected_ttest(output_folder, df):


if __name__ == "__main__":
    # output_folder = "/Users/taichi/Desktop/master_thesis/results/v3/"
    # # save_figs_tables(output_folder)
    # df = output_results_all_dimensions_kfold(difference_mode=True)
    # print(df.columns.values.tolist())

    # for index, row in df.iterrows():
    #     peak_end = row["Peak-End R^2"]    

    X, Y = data_loader("MD")
    