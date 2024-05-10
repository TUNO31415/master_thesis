from lstm_padding import lstm_with_padding_n_times_k_fold
from lstm_smart import lstm_smart_n_times_k_fold
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t
from utils import evaluation_metrics, data_loader, real_time_labels_distribution, split_train_test
import os

# paco_path = "/tudelft.net/staff-umbrella/tunoMSc2023/paco_dataset/"
paco_path = "/Users/taichi/Desktop/master_thesis/"
real_time_sis_folder_path = paco_path + "RealTimeSIS_v3_score_only/"
# real_time_sis_folder_path = paco_path + "RealTimeSIS_score_only/"
retrospective_sis_file_path = paco_path + "retrospective_sis.csv"

def smart_peak_end_rule(X_train, Y_train, X_test, Y_test):
    new_X_train = np.array([[max(x), x[-1]] for x in X_train])
    regressor = LinearRegression().fit(new_X_train, np.array(Y_train))
    print(f"peak weight : {regressor.coef_[0]}, end weight : {regressor.coef_[1]}, intercept : {regressor.intercept_}")
    new_X_test = np.array([[max(x), x[-1]] for x in X_test])
    Y_pred = regressor.predict(new_X_test)
    data_rows = [{
        "peak weight" : regressor.coef_[0],
        "end weight" : regressor.coef_[1],
        "intercept" : regressor.intercept_
    }]
    csv_file_path = "/Users/taichi/Desktop/master_thesis/results/v6/pe_regressor_weights_data.csv"
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        df = pd.DataFrame()
    df = df._append(data_rows, ignore_index=False)
    df.to_csv(csv_file_path, index=False)
    return evaluation_metrics(Y_test, Y_pred)

def peak_end_rule(X_train, Y_train, X_test, Y_test):
    Y_pred = [(max(x) + x[-1])/2 for x in X_test]
    return evaluation_metrics(Y_test, Y_pred)

def peak_only(X_train, Y_train, X_test, Y_test):
    Y_pred = [max(x) for x in X_test]
    return evaluation_metrics(Y_test, Y_pred)

def end_only(X_train, Y_train, X_test, Y_test):
    Y_pred = [x[-1] for x in X_test] 
    return evaluation_metrics(Y_test, Y_pred)

def base_line(X_train, Y_train, X_test, Y_test):
    Y_pred = [sum(x) / len(x) for x in X_test]
    return evaluation_metrics(Y_test, Y_pred)

def dummpy_regressor(X_train, Y_train, X_test, Y_test):
    mean = np.mean(np.array(Y_train))
    Y_pred = [mean] * len(Y_test)
    return evaluation_metrics(Y_test, Y_pred)

# model : peak_end_reg, peak_end, peak_only, end_only, base_line, lstm_pad, lstm_smart
# Results : List of eval_metrics [[r2_0, mse_0], [r2_1, mse_1], ..., [r2_100, mse_100]]
def output_all_results_all_dimension(model, output_path, n=10, repeat=10):

    functions = {
        "peak_end_reg" : smart_peak_end_rule,
        "peak_end" : peak_end_rule,
        "peak_only" : peak_only,
        "end_only" : end_only,
        "base_line" : base_line,
        "dummy" : dummpy_regressor
    }
    dimensions = ["MD", "CI", "FI", "IC", "P"]
    
    entries = []

    for d in dimensions:
        ress = []
        X, Y = data_loader(d)

        if model == "lstm_pad":
            entries.append(lstm_with_padding_n_times_k_fold(X, Y)) 
        elif model == "lstm_smart":
            entries.append(lstm_smart_n_times_k_fold(X, Y))
        else:
            for re in range(repeat):
                kf =KFold(n_splits=n, shuffle=True)
                for fold, (train_ids, test_ids) in enumerate(kf.split(X, Y)):
                    X_train, y_train, X_test, y_test = split_train_test(X, Y, train_ids, test_ids)
                    fc = functions[model]
                    eval = fc(X_train, y_train, X_test, y_test)
                    ress.append(eval)
            entries.append(ress)

    df = pd.DataFrame({
        "Dimension" : dimensions,
        "Model" : model,
        "Results" : entries
    })
    df.to_csv(output_path + f"{model}_all_results.csv")
    print(f"------ SAVED {model}_all_results.csv ------")

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

if __name__ == "__main__":
    output_folder = "/Users/taichi/Desktop/master_thesis/results/v6/"

    model_list = [
        "peak_end_reg",
        "peak_end",
        "peak_only",
        "end_only",
        "base_line",
        "dummy"
    ]

    for m in model_list:
        output_all_results_all_dimension(m, output_folder)