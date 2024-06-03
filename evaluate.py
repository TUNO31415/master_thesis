from lstm_padding import lstm_with_padding_n_times_k_fold
from lstm_smart import lstm_smart_n_times_k_fold
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from utils import evaluation_metrics, data_loader, split_train_test, retro_labels_distribution, real_time_labels_distribution, real_time_labels_distribution_new, convert_csv
import ast
from t_test import t_test
import os

# paco_path = "/tudelft.net/staff-umbrella/tunoMSc2023/paco_dataset/"
paco_path = "/Users/taichi/Desktop/master_thesis/"
# real_time_sis_folder_path = paco_path + "RealTimeSIS_v3_score_only/"
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
    csv_file_path = "/Users/taichi/Desktop/master_thesis/results/new_prompt_v1/pe_regressor_weights_data.csv"
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
def output_all_results_all_dimension(model, output_path, rt_sis_folder, retro_sis_file="/Users/taichi/Desktop/master_thesis/retrospective_sis.csv", n=10, repeat=10):

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    functions = {
        "peak_end_reg" : smart_peak_end_rule,
        "peak_end" : peak_end_rule,
        "peak_only" : peak_only,
        "end_only" : end_only,
        "base_line" : base_line,
        "dummy" : dummpy_regressor,
        "lstm_pad" : None
    }
    dimensions = ["MD", "CI", "FI", "IC", "P"]
    
    entries = []

    for d in dimensions:
        ress = []
        X, Y = data_loader(d, rt_sis_folder, retrospective_sis_file_path=retro_sis_file)

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

def process_all_results_all_dimension(output_path, model_list):
    # model_list = [
    #     "peak_end",
    #     "peak_end_reg",
    #     "peak_only",
    #     "end_only",
    #     "base_line",
    #     "dummy"
    # ]
    dimensions = ["MD", "CI", "FI", "IC", "P"]

    ds = []
    md = []
    r2_mean = []
    r2_std = []
    mse_mean = []
    mse_std = []
    for model in model_list:
        df = pd.read_csv(output_path + f"{model}_all_results.csv")
        for i, row in df.iterrows():
            ds.append(row["Dimension"])
            md.append(model)
            results = ast.literal_eval(row["Results"])
            r2s = np.array([float(a[0]) for a in results])
            mses = np.array([float(a[1]) for a in results])
            r2_mean.append(np.mean(r2s))
            r2_std.append(np.std(r2s))
            mse_mean.append(np.mean(mses))
            mse_std.append(np.std(mses))

    df = pd.DataFrame({
        "Dimension" : ds,
        "Model" : md,
        "r2_mean" : r2_mean,
        "r2_std" : r2_std,
        "mse_mean" : mse_mean,
        "mse_std" : mse_std
    })
    df.sort_values("Dimension")
    df.to_csv(output_path + "processed_results_all.csv")
    print(f"----- SAVED {output_path}processed_results.csv ------")

def replace_substring(s):
    return s.replace("lstm_smart", "lstm_length_varying")

def plot_error_plot(output_folder):
    dimensions = ["MD", "CI", "FI", "IC", "P"]

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    df = pd.read_csv(output_folder + "processed_results_all.csv")

    for d in dimensions:
        rows = df[(df["Dimension"] == d)]
        r2_means = np.array(rows["r2_mean"])
        r2_stds = np.array(rows["r2_std"])
        # mse_means = np.array(rows["mse_mean"])
        # mse_stds = np.array(rows["mse_std"])
        models = np.array(rows["Model"])
        vectorized_replace = np.vectorize(replace_substring)
        models = vectorized_replace(models)

        if d == "CI":
            plt.errorbar(models, r2_means, r2_stds, linestyle='None', marker='^', color='tab:blue', ecolor='tab:cyan', capsize=5)
            plt.xticks(models, models, rotation=25)
            plt.ylim(-18, 0)
            plt.gcf().set_size_inches(7, 5)  # Set custom figure width and height
            plt.title(f"{d}_r2")
            plt.grid(True, axis='y', linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.savefig(output_folder + f"{d}_R^2_result.png", dpi=300)
            plt.close()
        else:
            plt.errorbar(models, r2_means, r2_stds, linestyle='None', marker='^', color='tab:blue', ecolor='tab:cyan', capsize=5)
            plt.xticks(models, models, rotation=25)
            plt.ylim(-1.7, 0)
            plt.gcf().set_size_inches(7, 5)  # Set custom figure width and height
            plt.title(f"{d}_r2")
            plt.grid(True, axis='y', linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.savefig(output_folder + f"{d}_R^2_result.png", dpi=300)
            plt.close()

def plot_error_plot_selective(output_folder):
    dimensions = ["MD", "CI", "FI", "IC", "P"]

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    df = pd.read_csv(output_folder + "processed_results_all.csv")

    for d in dimensions:
        desired_models = ["peak_end_reg", "lstm_pad", "lstm_smart", "dummy"]
        rows = df[(df["Dimension"] == d) & (df["Model"].isin(desired_models))]
        r2_means = np.array(rows["r2_mean"])
        r2_stds = np.array(rows["r2_std"])
        # mse_means = np.array(rows["mse_mean"])
        # mse_stds = np.array(rows["mse_std"])
        models = np.array(rows["Model"])
        vectorized_replace = np.vectorize(replace_substring)
        models = vectorized_replace(models)
        plt.errorbar(models, r2_means, r2_stds, linestyle='None', marker='^', color='tab:blue', ecolor='tab:cyan', capsize=5)
        plt.xticks(models, models, rotation=25)
        plt.ylim(-0.8, 0.3)
        plt.gcf().set_size_inches(7, 5)  # Set custom figure width and height
        plt.title(f"{d}_r2")
        plt.grid(True, axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(output_folder + f"{d}_R^2_result_selective.png", dpi=300)
        plt.close()

if __name__ == "__main__":
    # output_folder = "/Users/taichi/Desktop/master_thesis/results/new_prompt_v1/"
    # output_folder = "/Users/taichi/Desktop/master_thesis/results/without_number_prompt/"
    output_folder_per_question = "/Users/taichi/Desktop/master_thesis/results/per_question_promt/"
    score_only_per_question_folder = "/Users/taichi/Desktop/master_thesis/RealTimeSIS_per_question_v1/score_only/"

    output_folder_without_number = "/Users/taichi/Desktop/master_thesis/results/without_number_prompt/"
    score_only_without_number_folder = "/Users/taichi/Desktop/master_thesis/RealTimeSIS_without_number/score_only/"

    model_list = [
        "peak_end_reg",
        "peak_end",
        "peak_only",
        "end_only",
        "base_line",
        "dummy"
    ]
    # for m in model_list:
    #     output_all_results_all_dimension(m, output_folder_per_question, score_only_per_question_folder)

    model_list = [
        "peak_end_reg",
        "peak_end",
        "peak_only",
        "end_only",
        "lstm_pad",
        "lstm_smart",
        "base_line",
        "dummy"
    ]

    pairs = [
        ["peak_end", "dummy"],
        ["peak_end_reg", "dummy"],
        ["peak_end", "base_line"],
        ["peak_end_reg", "base_line"],
        ["peak_end", "end_only"],
        ["peak_end", "peak_only"],
        ["lstm_pad", "dummy"],
        ["lstm_smart", "dummy"],
        ["lstm_pad", "base_line"],
        ["lstm_smart", "base_line"]
    ]


    # # Example usage
    # input_file = "/Users/taichi/Desktop/lstm_pad_all_results 2.csv"
    # output_file = output_folder + "/lstm_pad_all_results.csv"
    # convert_csv(input_file, output_file)

    # input_file = "/Users/taichi/Desktop/lstm_smart_all_results 2.csv"
    # output_file = output_folder + "/lstm_smart_all_results.csv"
    # convert_csv(input_file, output_file)

    # process_all_results_all_dimension(output_folder, model_list)
    # # retro_labels_distribution(output_folder)
    # plot_error_plot(output_folder)
    # # plot_error_plot_selective(output_folder)
    # t_test(output_folder, pairs)
    real_time_labels_distribution_new(output_folder_per_question, rt_folder=score_only_per_question_folder)