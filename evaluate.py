from lstm_padding import lstm_with_padding_n_times_k_fold
from lstm_smart import lstm_smart_n_times_k_fold
import pandas as pd
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t
from utils import evaluation_metrics, data_loader, real_time_labels_distribution
import os

# paco_path = "/tudelft.net/staff-umbrella/tunoMSc2023/paco_dataset/"
paco_path = "/Users/taichi/Desktop/master_thesis/"
real_time_sis_folder_path = paco_path + "RealTimeSIS_v3_score_only/"
# real_time_sis_folder_path = paco_path + "RealTimeSIS_score_only/"
retrospective_sis_file_path = paco_path + "retrospective_sis.csv"

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

def print_evaluation_result(model_name, dimension, eval_list):
    print(f"{model_name} {dimension} | R2 : {eval_list[0]} | MSE : {eval_list[1]}")

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
                "Dimension" : d,
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
            entry_r2 = {
                    "Dimension" : d, 
                    "Peak-End R^2 mean" : np.mean(results_pe_r), 
                    "Peak-End R^2 std" : np.std(results_pe_r), 
                    "Peak-Only R^2 mean" : np.mean(results_p_r), 
                    "Peak-Only R^2 std" : np.std(results_p_r), 
                    "End-Only R^2 mean" : np.mean(results_e_r),
                    "End-Only R^2 std" : np.std(results_e_r),
                    "Base R^2 mean" : np.mean(results_base_r),
                    "Base R^2 std" : np.std(results_base_r),
                    }

            entry_mse = {
                    "Dimension" : d, 
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
    # df_r2, df_mse = output_results_all_dimensions_kfold()
    # df_r2.to_csv(output_folder + f"result_r2.csv")
    # df_mse.to_csv(output_folder + f"result_MSE.csv")
    df_r2 = pd.read_csv(output_folder + f"result_r2.csv", delimiter=';')
    df_mse = pd.read_csv(output_folder + f"result_MSE.csv", delimiter=';')
    x_axis = np.array(["Peak-End", "Peak-Only", "End-Only", "Baseline", "LSTM-padding"])

    print(df_mse.columns)
    for index, row in df_mse.iterrows():
        y_axis = np.array([row["Peak-End MSE mean"], row["Peak-Only MSE mean"], row["End-Only MSE mean"], row["Base MSE mean"], row["LSTM padding mean"]])
        err = np.array([row["Peak-End MSE std"], row["Peak-Only MSE std"], row["End-Only MSE std"], row["Base MSE std"], row["LSTM padding std"]])
        err = err / 10
        plt.errorbar(x_axis, y_axis, err, linestyle='None', marker='^')
        plt.title(f"{row["Dimension"]}_MSE_result")
        plt.savefig(output_folder + f"{row["Dimension"]}_MSE_result.png")
        plt.close()

        
    for index, row in df_r2.iterrows():
        y_axis = np.array([row[f"Peak-End R^2 mean"], row["Peak-Only R^2 mean"], row["End-Only R^2 mean"], row["Base R^2 mean"], row["LSTM padding R^2 mean"]])
        err = np.array([row["Peak-End R^2 std"], row["Peak-Only R^2 std"], row["End-Only R^2 std"], row["Base R^2 std"], row["LSTM padding R^2 std"]])
        err = err / 10
        plt.errorbar(x_axis, y_axis, err, linestyle='None', marker='^')
        plt.title(f"{row["Dimension"]}_R^2_result")
        plt.savefig(output_folder + f"{row["Dimension"]}_R^2_result.png")
        plt.close()

# def save_corrected_ttest(output_folder, df):


if __name__ == "__main__":
    # save_figs_tables("/Users/taichi/Desktop/master_thesis/results/v4/")
    # df = output_results_all_dimensions_kfold(difference_mode=True)
    # output_folder = "/Users/taichi/Desktop/master_thesis/full_results/"

    # if not os.path.exists(output_folder):
    #     os.makedirs(output_folder)

    # df.to_csv(output_folder + "pe_full.csv")
    # print("DONE")

    real_time_labels_distribution("/Users/taichi/Desktop/master_thesis/estimated_real-time_sis_histograms/")
    print("DONE")

    # # save_figs_tables(output_folder)
    # df = output_results_all_dimensions_kfold(difference_mode=True)
    # print(df.columns.values.tolist())

    # for index, row in df.iterrows():
    #     peak_end = row["Peak-End R^2"]    

    # dimensions = ["MD", "CI", "FI", "IC", "P"]
    # r2_entries = []
    # mse_entries = []
    # output_folder = paco_path + "result_lstm/"

    # if not os.path.exists(output_folder):
    #     os.makedirs(output_folder)

    # labels = ["lstm padding", "lstm lengths varying"]
    # for d in dimensions:
    #     print(f"---- {d} DIMENSION START ----")
    #     X, Y = data_loader(d)
    #     res = lstm_smart_n_times_k_fold(X, Y)
    #     print("smart DONE")

    #     labels = ["lstm padding", "lstm lengths varying"]

        
    #     r2 = np.array([a[0] for a in res])
    #     mse = np.array([a[1] for a in res])

    #     r2_mean = np.mean(r2)
    #     r2_std = np.std(r2)
    #     mse_mean = np.mean(mse)
    #     mse_std = np.std(mse)

    #     r2_entry = {"Dimension" : d, "Model" : labels[1], "mean" : r2_mean, "std" : r2_std}
    #     mse_entry = {"Dimension" : d, "Model" : labels[1], "mean" : mse_mean, "std" : mse_std}

    #     r2_entries.append(r2_entry)
    #     mse_entries.append(mse_entry)

    # df_r2 = pd.DataFrame(r2_entries)
    # df_r2.to_csv(output_folder + "lstm_smart_results_r2.csv")

    # df_mse = pd.DataFrame(mse_entries)
    # df_mse.to_csv(output_folder + "lstm_smart_results_mse.csv")


    # for d in dimensions:
    #     print(f"---- {d} DIMENSION START ----")
    #     X, Y = data_loader(d)
    #     res = lstm_with_padding_n_times_k_fold(X, Y)
    #     print("padding DONE")

    #     r2 = np.array([a[0] for a in res])
    #     mse = np.array([a[1] for a in res])

    #     r2_mean = np.mean(r2)
    #     r2_std = np.std(r2)
    #     mse_mean = np.mean(mse)
    #     mse_std = np.std(mse)

    #     r2_entry = {"Dimension" : d, "Model" : labels[0], "mean" : r2_mean, "std" : r2_std}
    #     mse_entry = {"Dimension" : d, "Model" : labels[0], "mean" : mse_mean, "std" : mse_std}

    #     r2_entries.append(r2_entry)
    #     mse_entries.append(mse_entry)
        
    
    # df_r2 = pd.DataFrame(r2_entries)
    # df_r2.to_csv(output_folder + "lstm_padding_results_r2.csv")

    # df_mse = pd.DataFrame(mse_entries)
    # df_mse.to_csv(output_folder + "lstm_padding_results_mse.csv")


    # dimensions = ["MD", "CI", "FI", "IC", "P"]
    # output_folder = paco_path + "result_lstm/"

    # if not os.path.exists(output_folder):
    #     os.makedirs(output_folder)

    # r2_results = []
    # mse_results = []
    # for d in dimensions:
    #     print(f"---- {d} DIMENSION START ----")
    #     X, Y = data_loader(d)
    #     res = lstm_smart_n_times_k_fold(X, Y)
    #     print("smart DONE")

        
    #     r2 = [a[0] for a in res]
    #     mse = [a[1] for a in res]
    #     r2_results.append(r2)
    #     mse_results.append(mse)
    
    # df = pd.DataFrame({
    #     "Dimension" : dimensions,
    #     "r2 results" : r2_results,
    #     "mse results" : mse_results
    # })

    # df.to_csv(output_folder + "lstm_smart_full_results.csv", index=False)