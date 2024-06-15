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
from collections import Counter

# paco_path = "/tudelft.net/staff-umbrella/tunoMSc2023/paco_dataset/"
paco_path = "/Users/taichi/Desktop/master_thesis/"
# real_time_sis_folder_path = paco_path + "RealTimeSIS_v3_score_only/"
# real_time_sis_folder_path = paco_path + "RealTimeSIS_score_only/"
retrospective_sis_file_path = paco_path + "retrospective_sis.csv"

def smart_peak_end_rule(X_train, Y_train, X_test, Y_test):
    new_X_train = np.array([[max(x), x[-1]] for x in X_train])
    regressor = LinearRegression().fit(new_X_train, np.array(Y_train))
    # print(f"peak weight : {regressor.coef_[0]}, end weight : {regressor.coef_[1]}, intercept : {regressor.intercept_}")
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

        # if d == "CI":
        #     fig = plt.figure(figsize=(10, 7))
        #     bax = brokenaxes(ylims=((-9.5, -3.5), (-1.0 , 0.3)), hspace=0.05)

        #     # Plot data with error bars
        #     bax.errorbar(models, r2_means, yerr=r2_stds, linestyle='None', marker='^', color='tab:blue', ecolor='tab:cyan', capsize=4)

        #     # Align x-ticks
        #     for ax in bax.axs:
        #         ax.set_xticks(range(len(models)))
        #         ax.set_xticklabels(models, rotation=20)
        #         # Make spines visible and set their properties for border effect
        #         for spine in ax.spines.values():
        #             spine.set_visible(True)
        #             spine.set_color('black')
        #             spine.set_linewidth(1.5)

        #     # Set the title for the figure
        #     fig.suptitle(f"{d}_r2")

        #     # Set grid on y-axis
        #     bax.grid(True, axis='y', linestyle='--', alpha=0.5)

        #     # Save the figure
        #     plt.savefig(output_folder + f"{d}_R^2_result.png", dpi=300)

        #     # Close the plot
        #     plt.close()

        if d == "P":
            fig = plt.figure(figsize=(10, 7))
            bax = brokenaxes(ylims=((-6.5, -3.3), (-2.8 , 0.3)), hspace=0.05)

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
            fig = plt.figure(figsize=(10, 7))
            bax = brokenaxes(hspace=0.05)
            bax.set_ylim(-1.5, 0.3)
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

    max_diff_instance = ""
    min_diff_instance = ""
    max_tmp = 0
    min_tmp = 10000
    max_list_retro = []
    max_list_sum = []
    min_list_retro = []
    min_list_sum = []
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

        total_diff = 0
        tmp_retro = []
        tmp_sum = []
        for d in dimensions:
            retro_val_tmp = retro_row.iloc[0][d]
            sum_val_tmp = df_summary.iloc[0][d]
            retro_labels[d].append(retro_val_tmp)
            sum_labels[d].append(sum_val_tmp)
            total_diff += abs(retro_val_tmp - sum_val_tmp)
            tmp_retro.append(retro_val_tmp)
            tmp_sum.append(sum_val_tmp)


        if total_diff > max_tmp:
            max_tmp = total_diff
            max_diff_instance = sum_csv
            max_list_retro = tmp_retro
            max_list_sum = tmp_sum
        
        if total_diff < min_tmp:
            min_tmp = total_diff
            min_diff_instance = sum_csv
            min_list_retro = tmp_retro
            min_list_sum = tmp_sum

    print(f"MAX INSTANCE : {max_diff_instance} : ({max_tmp})")
    print(f"MAX EST SUMMARY : {max_list_sum}")
    print(f"MAX RETRO GROUND TRUTH : {max_list_retro}")
    print(f"MIN INSTANCE : {min_diff_instance} : ({min_tmp})")
    print(f"MIN EST SUMMARY : {min_list_sum}")
    print(f"MIN RETRO GROUND TRUTH : {min_list_retro}")


    for d in dimensions:
        mae = mean_absolute_error(retro_labels[d], sum_labels[d])
        r = np.corrcoef(retro_labels[d], sum_labels[d])
        r2 = r2_score(retro_labels[d], sum_labels[d])

        # print(f"=== Dimension {d} ===")
        # print(f"MAE : {mae}")
        # print(f"R^2 : {r2}")
        # print(f"Correlation : {r}")

def compare_summary_and_retro_fold(summary_folder_path, retro_csv_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    df_retro = pd.read_csv(retro_csv_path)

    # total_num = 0
    # class_avg = class_average_retrospective(retro_csv_path)

    retro_labels = {"MD" : [],"CI" : [],"FI" : [],"IC" : [],"P" : []}
    sum_labels = {"MD" : [],"CI" : [],"FI" : [],"IC" : [],"P" : []}

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
    
    
    output_list = []

    for d in dimensions:
        row = {
            "dimension" : d,
            "retro_mae" : [],
            "sum_mae" : [],
            "retro_r2" : [],
            "sum_r2" : [],
        }

        sum_current = np.array(sum_labels[d])
        retro_current = np.array(retro_labels[d])

        kf = KFold(n_splits=10)
        for train, test in kf.split(sum_labels[d]):
            sum_train, sum_test, retro_train, retro_test  = sum_current[train], sum_current[test], retro_current[train], retro_current[test]
            mean = np.mean(np.array(retro_train))
            retro_values = [mean] * len(retro_test)
            sum_values = sum_test
            retro_r2 = r2_score(retro_values, retro_test)
            sum_r2 = r2_score(sum_values, retro_test)
            retro_mae = mean_absolute_error(retro_values, retro_test)
            sum_mae = mean_absolute_error(sum_values, retro_test)
            row["retro_mae"].append(retro_mae)
            row["sum_mae"].append(sum_mae)
            row["retro_r2"].append(retro_r2)
            row["sum_r2"].append(sum_r2)

        output_list.append(row)

    output_df = pd.DataFrame(output_list)
    output_df.to_csv(output_path + "fold_retro_sum_evaluation.csv")

    aggregated_df = []
    for i, d in enumerate(dimensions):
        aggregated_df.append({
            "Dimension" : d,
            "Folded Retrospective R_2 mean" : float(np.mean(output_list[i]["retro_r2"])),
            "Folded Retrospective R_2 std" : float(np.std(output_list[i]["retro_r2"])),
            "Estimated Summary Evaluation R_2 mean" : float(np.mean(output_list[i]["sum_r2"])),
            "Estimated Summary Evaluation R_2 std" : float(np.std(output_list[i]["sum_r2"])),
            "Folded Retrospective MAE mean" : float(np.mean(output_list[i]["retro_mae"])),
            "Folded Retrospective MAE std" : float(np.std(output_list[i]["retro_mae"])),
            "Estimated Summary Evaluation MAE mean" : float(np.mean(output_list[i]["sum_mae"])),
            "Estimated Summary Evaluation MAE std" : float(np.std(output_list[i]["sum_mae"]))
        })

    aggregated_df = pd.DataFrame(aggregated_df)
    aggregated_df.to_csv(output_path + "aggregated_table.csv")

def plot_scatter_sum_retro(retro_csv_path, sum_csv_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    retro_csv_df = pd.read_csv(retro_csv_path)
    sum_csv_df = pd.read_csv(sum_csv_path)

    dimensions = ["MD", "CI", "FI", "IC", "P"]
    for d in dimensions:
        x = []
        y = []
        
        for index, sum_row in sum_csv_df.iterrows():
            batch_id = sum_row["BatchNum"]
            selfPID = sum_row["selfPID"]
            otherPID = sum_row["otherPID"]
            
            x_value = float(sum_row[d])
            x.append(x_value)
            
            retro_row = retro_csv_df.loc[(retro_csv_df["BatchNum"] == batch_id) & 
                                        (retro_csv_df["selfPID"] == selfPID) & 
                                        (retro_csv_df["otherPID"] == otherPID)]
            
            if len(retro_row) == 0:
                continue
            elif len(retro_row) > 1:
                retro_row = retro_row.iloc[0]
            
            y_value = float(retro_row[d])
            y.append(y_value)
        
        # Count occurrences of each (x, y) pair
        counts = Counter(zip(x, y))
        
        # Separate the x, y values and their corresponding sizes
        unique_x = []
        unique_y = []
        sizes = []
        
        for (x_val, y_val), count in counts.items():
            unique_x.append(x_val)
            unique_y.append(y_val)
            sizes.append(count * 5)  # Adjust the multiplier to control point size
        
        # Create scatter plot with sizes based on counts
        # Plot y=x line
        plt.plot([0, 5], [0, 5], linestyle='--', color='gray')
        # Add grid lines
        plt.grid(True)
        plt.scatter(unique_x, unique_y, s=sizes, alpha=0.5)
        plt.xlim((1.0, 5.0))
        plt.ylim((1.0, 5.0))
        plt.xlabel(f"Retrospective evaluation of {d}")
        plt.ylabel(f"Estimated summary evaluation of {d}")
        plt.title(f"Scatter Plot of {d}")
        plt.savefig(output_path + f"scatterplot_{d}.png")
        plt.close()

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
    output_folder_v2 = "/Users/taichi/Desktop/master_thesis/results/final/numerical_without_context/"
    score_only_output_folder_v2 = "/Users/taichi/Desktop/master_thesis/RealTimeSIS_newprompt/score_only/"
    
    # output_folder_per_question = "/Users/taichi/Desktop/master_thesis/results/per_question_promt/"
    # score_only_per_question_folder = "/Users/taichi/Desktop/master_thesis/RealTimeSIS_per_question_v1/score_only/"

    # output_folder_without_number = "/Users/taichi/Desktop/master_thesis/results/without_number_prompt/"
    # score_only_without_number_folder = "/Users/taichi/Desktop/master_thesis/RealTimeSIS_without_number/score_only/"

    # output_folder_with_context = "/Users/taichi/Desktop/master_thesis/results/with_context/"
    # score_only_with_context = "/Users/taichi/Desktop/master_thesis/RealTimeSIS_with_context/score_only/"

    output_folder_with_context = "/Users/taichi/Desktop/master_thesis/results/final/with_context/"
    score_only_with_context = "/Users/taichi/Desktop/master_thesis/RealTimeSIS_with_context_v2/score_only/"

    model_list = [
        "peak_end_reg",
        "peak_end",
        "peak_only",
        "end_only",
        "base_line",
        "dummy"
    ]
    # for m in model_list:
    #     output_all_results_all_dimension(m, output_folder_with_context, score_only_with_context)

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

    # process_all_results_all_dimension(output_folder_with_context, model_list)
    # # # retro_labels_distribution(output_folder)
    plot_error_plot_brokenaxes(output_folder_with_context)
    # plot_error_plot_selective(output_folder_without_number)
    # t_test(output_folder_with_context, pairs)
    # real_time_labels_distribution_new(output_folder_v2, "numerical_output", rt_folder=score_only_output_folder_v2)
    # real_time_labels_distribution_new(output_folder_with_context, "final", rt_folder=score_only_with_context)
    # real_time_labels_distribution_new(output_folder_without_number, "without_number", rt_folder=score_only_without_number_folder)
    # real_time_labels_distribution_new(output_folder_with_context, "with_context", rt_folder=score_only_with_context)
    # plot_scatter_sum_retro("/Users/taichi/Desktop/master_thesis/retrospective_sis.csv", "/Users/taichi/Desktop/master_thesis/estimated_summary_sis.csv", "/Users/taichi/Desktop/master_thesis/results/sum_retro_eval/")
    
    # compare_summary_and_retro_fold("/Users/taichi/Desktop/master_thesis/RealTimeSIS_summary_label/score_only/", "/Users/taichi/Desktop/master_thesis/retrospective_sis.csv", "/Users/taichi/Desktop/master_thesis/results/final/summary/")