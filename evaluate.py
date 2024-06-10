from lstm_padding import lstm_with_padding_n_times_k_fold
from lstm_smart import lstm_smart_n_times_k_fold
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
from utils import evaluation_metrics, data_loader, split_train_test, retro_labels_distribution, real_time_labels_distribution, real_time_labels_distribution_new, convert_csv
import ast
from t_test import t_test
import os
from brokenaxes import brokenaxes

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
            plt.ylim(-18, 0.3)
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

def plot_error_plot_brokenaxes(output_folder):
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
            # fig = plt.figure()
            # bax = brokenaxes(ylims=((-10, -3),(-1.7, 0)), hspace=0.05)
            # bax.errorbar(models, r2_means, yerr=r2_stds, linestyle='None', marker='^', color='tab:blue', ecolor='tab:cyan', capsize=5)

            # # Adjust x-ticks manually
            # bax.set_xticks(range(len(models)))
            # bax.set_xticklabels(models, rotation=25)

            # # Set the title for the figure
            # fig.suptitle(f"{d}_r2")

            # # Set grid on y-axis
            # bax.grid(True, axis='y', linestyle='--', alpha=0.5)

            # # Save the figure
            # plt.savefig(output_folder + f"{d}_R^2_result.png", dpi=300)

            # # Close the plot
            # plt.close()

            fig = plt.figure(figsize=(10, 7))
            bax = brokenaxes(ylims=((-9.5, -2.5), (-1.7, 0.1)), hspace=0.05)

            # Plot data with error bars
            bax.errorbar(models, r2_means, yerr=r2_stds, linestyle='None', marker='^', color='tab:blue', ecolor='tab:cyan', capsize=4)

            # Align x-ticks
            for ax in bax.axs:
                ax.set_xticks(range(len(models)))
                ax.set_xticklabels(models, rotation=20)
                # Make spines visible and set their properties for border effect
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_color('black')
                    spine.set_linewidth(1.5)

            # Set the title for the figure
            fig.suptitle(f"{d}_r2")

            # Set grid on y-axis
            bax.grid(True, axis='y', linestyle='--', alpha=0.5)

            # Save the figure
            plt.savefig(output_folder + f"{d}_R^2_result.png", dpi=300)

            # Close the plot
            plt.close()
        else:
            # plt.errorbar(models, r2_means, r2_stds, linestyle='None', marker='^', color='tab:blue', ecolor='tab:cyan', capsize=4)
            # plt.xticks(models, models, rotation=20)
            # plt.ylim(-1.7, 0)
            # plt.gcf().set_size_inches(10, 7)  # Set custom figure width and height
            # plt.title(f"{d}_r2")
            # plt.grid(True, axis='y', linestyle='--', alpha=0.5)
            # # plt.tight_layout()
            # plt.savefig(output_folder + f"{d}_R^2_result.png", dpi=300)
            # plt.close()
            fig = plt.figure(figsize=(10, 7))
            bax = brokenaxes(hspace=0.05)
            bax.set_ylim(-1.7, 0.1)
            # Plot data with error bars
            bax.errorbar(models, r2_means, yerr=r2_stds, linestyle='None', marker='^', color='tab:blue', ecolor='tab:cyan', capsize=4)

            # Align x-ticks
            for ax in bax.axs:
                ax.set_xticks(range(len(models)))
                ax.set_xticklabels(models, rotation=20)
                # Make spines visible and set their properties for border effect
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_color('black')
                    spine.set_linewidth(1.5)

            # Set the title for the figure
            fig.suptitle(f"{d}_r2")

            # Set grid on y-axis
            bax.grid(True, axis='y', linestyle='--', alpha=0.5)

            # Save the figure
            plt.savefig(output_folder + f"{d}_R^2_result.png", dpi=300)

            # Close the plot
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

def compare_summary_and_retro(summary_folder_path, retro_csv_path):

    df_retro = pd.read_csv(retro_csv_path)
    mae_sum = {
        "MD" : 0,
        "CI" : 0,
        "FI" : 0,
        "IC" : 0,
        "P" : 0
    }

    mae_avg = {
        "MD" : 0,
        "CI" : 0,
        "FI" : 0,
        "IC" : 0,
        "P" : 0
    }

    accuracy_sum = {
        "MD" : 0,
        "CI" : 0,
        "FI" : 0,
        "IC" : 0,
        "P" : 0
    }

    total_num = 0
    class_avg = class_average_retrospective(retro_csv_path)

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

# score_only_summary_SIS_PID_28_Batch_4_PID_28_PID_33.csv
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

        for d in dimensions:
            retro_labels[d].append(retro_row.iloc[0][d])
            sum_labels[d].append(df_summary.iloc[0][d])

    for d in dimensions:
        mae = mean_absolute_error(retro_labels[d], sum_labels[d])
        r = np.corrcoef(retro_labels[d], sum_labels[d])
        r2 = r2_score(retro_labels[d], sum_labels[d])

        print(f"=== Dimension {d} ===")
        print(f"MAE : {mae}")
        print(f"R^2 : {r2}")
        print(f"Correlation : {r}")

    # print(f"TOTAL NUMBER OF PREDICTIONS : {total_num}")
    # print(f"CLASS AVERAGE : {class_avg}")
    # print(f"MAE of summary estimation : {mae_sum}")
    # print(f"MAE of class average : {mae_avg}")
    # print(f"Accuracy of summary estimation : {accuracy_sum}")

def class_average_retrospective(retro_csv_path):
    
    average = {
        "MD" : 0,
        "CI" : 0,
        "FI" : 0,
        "IC" : 0,
        "P" : 0
    }

    df = pd.read_csv(retro_csv_path)
    dimensions = ["MD", "CI", "FI", "IC", "P"]

    total_num = 0
    
    for index, row in df.iterrows():
        for d in dimensions:
            average[d] += row[d]
        total_num += 1

    for i in average:
        average[i] = average[i] / total_num

    return average
    
if __name__ == "__main__":
    output_folder = "/Users/taichi/Desktop/master_thesis/results/new_prompt_v2/"
    # output_folder = "/Users/taichi/Desktop/master_thesis/results/without_number_prompt/"
    output_folder_per_question = "/Users/taichi/Desktop/master_thesis/results/per_question_promt/"
    score_only_per_question_folder = "/Users/taichi/Desktop/master_thesis/RealTimeSIS_per_question_v1/score_only/"

    output_folder_without_number = "/Users/taichi/Desktop/master_thesis/results/without_number_prompt/"
    score_only_without_number_folder = "/Users/taichi/Desktop/master_thesis/RealTimeSIS_without_number/score_only/"

    output_folder_with_context = "/Users/taichi/Desktop/master_thesis/results/with_context/"
    score_only_with_context = "/Users/taichi/Desktop/master_thesis/RealTimeSIS_with_context/score_only/"

    model_list = [
        "peak_end_reg",
        "peak_end",
        "peak_only",
        "end_only",
        "base_line",
        "dummy"
    ]
    for m in model_list:
        output_all_results_all_dimension(m, output_folder_with_context, score_only_with_context)

    # model_list = [
    #     "peak_end_reg",
    #     "peak_end",
    #     "peak_only",
    #     "end_only",
    #     "lstm_pad",
    #     "lstm_smart",
    #     "base_line",
    #     "dummy"
    # ]

    # pairs = [
    #     ["peak_end", "dummy"],
    #     ["peak_end_reg", "dummy"],
    #     ["peak_end", "base_line"],
    #     ["peak_end_reg", "base_line"],
    #     ["peak_end", "end_only"],
    #     ["peak_end", "peak_only"],
    #     ["lstm_pad", "dummy"],
    #     ["lstm_smart", "dummy"],
    #     ["lstm_pad", "base_line"],
    #     ["lstm_smart", "base_line"]
    # ]


    # # Example usage
    # input_file = "/Users/taichi/Desktop/lstm_pad_all_results 2.csv"
    # output_file = output_folder + "/lstm_pad_all_results.csv"
    # convert_csv(input_file, output_file)

    # input_file = "/Users/taichi/Desktop/lstm_smart_all_results 2.csv"
    # output_file = output_folder + "/lstm_smart_all_results.csv"
    # convert_csv(input_file, output_file)

    # process_all_results_all_dimension(output_folder_without_number, model_list)
    # # # retro_labels_distribution(output_folder)
    # plot_error_plot_brokenaxes(output_folder)
    # plot_error_plot_selective(output_folder_without_number)
    # t_test(output_folder_without_number, pairs)
    # real_time_labels_distribution_new(output_folder_without_number, rt_folder=score_only_without_number_folder)
    # real_time_labels_distribution_new(output_folder_per_question, rt_folder=score_only_per_question_folder)
    # 
    
