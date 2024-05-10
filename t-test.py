from scipy.stats import ttest_ind
import pandas as pd
from evaluate import output_results_all_dimensions_kfold

def main():
    df = output_results_all_dimensions_kfold(difference_mode=True)
    dimensions = ["MD", "CI", "FI", "IC", "P"]

#     entry = {
#     "Dimension" : d,
#     "Peak-End R^2" : results_pe_r,
#     "Peak-Only R^2" : results_p_r,
#     "End-Only R^2" : results_e_r,
#     "Base R^2" : results_base_r,
#     "Peak-End MSE" : results_pe_m,
#     "Peak-Only MSE" : results_p_m,
#     "End-Only MSE" : results_e_m,
#     "Base MSE" : results_base_m,
# }
    entries = []

    for id, row in df.iterrows():
        results_pe_mse = row["Peak-End MSE"]
        results_p_mse = row["Peak-Only MSE"]
        results_e_mse = row["End-Only MSE"]
        results_base_mse = row["Base MSE"]

        ttest_res = ttest_ind(results_pe_mse, results_p_mse, equal_var=False)

        entry = {
            "Dimension" : row["Dimension"],
            "Model 1" : "Peak End",
            "Model 2" : "Peak Only",
            "p-value" : ttest_res.pvalue,
            "t-statistics" : ttest_res.statistic,
            "Degree of Freedom" : ttest_res.df,
            "Confidence Interval low" : ttest_res.confidence_interval().low,
            "Confidence Interval high" : ttest_res.confidence_interval().high,
        }
        entries.append(entry)

        ttest_res = ttest_ind(results_pe_mse, results_e_mse, equal_var=False)

        entry = {
            "Dimension" : row["Dimension"],
            "Model 1" : "Peak End",
            "Model 2" : "End Only",
            "p-value" : ttest_res.pvalue,
            "t-statistics" : ttest_res.statistic,
            "Degree of Freedom" : ttest_res.df,
            "Confidence Interval low" : ttest_res.confidence_interval().low,
            "Confidence Interval high" : ttest_res.confidence_interval().high,
        }
        entries.append(entry)

        ttest_res = ttest_ind(results_pe_mse, results_base_mse, equal_var=False)

        entry = {
            "Dimension" : row["Dimension"],
            "Model 1" : "Peak End",
            "Model 2" : "Baseline",
            "p-value" : ttest_res.pvalue,
            "t-statistics" : ttest_res.statistic,
            "Degree of Freedom" : ttest_res.df,
            "Confidence Interval low" : ttest_res.confidence_interval().low,
            "Confidence Interval high" : ttest_res.confidence_interval().high,
        }
        entries.append(entry)
    
    df_res = pd.DataFrame(entries)
    df_res.to_csv("/Users/taichi/Desktop/master_thesis/results/t-test_MSE_results.csv")
    print("DONE")

if __name__ == "__main__":
    main()
