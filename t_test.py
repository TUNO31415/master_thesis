from scipy.stats import ttest_ind
import pandas as pd
import ast

def t_test(output_folder, pairs):
    dimensions = ["MD", "CI", "FI", "IC", "P"]

    for d in dimensions:
        entries = []
        for p in pairs:
            m_1 = p[0]
            m_2 = p[1]
            df_m1 = pd.read_csv(output_folder + f"{m_1}_all_results.csv")
            df_m2 = pd.read_csv(output_folder + f"{m_2}_all_results.csv")
            results_m1 = ast.literal_eval(df_m1[(df_m1["Dimension"] == d)]["Results"].tolist()[0])
            results_m2 = ast.literal_eval(df_m2[(df_m2["Dimension"] == d)]["Results"].tolist()[0])
            r2_m1 = [a[0] for a in results_m1]
            r2_m2 = [a[0] for a in results_m2]
            ttest_res = ttest_ind(r2_m1, r2_m2, equal_var=False)

            entries.append(
                {
                    "Dimension" : d,
                    "Model 1" : m_1,
                    "Model 2" : m_2,
                    "p-value" : ttest_res.pvalue,
                    "t-statistics" : ttest_res.statistic,
                    "Degree of Freedom" : ttest_res.df,
                    "Confidence Interval low" : ttest_res.confidence_interval().low,
                    "Confidence Interval high" : ttest_res.confidence_interval().high,
                }
            )
        
        df_res = pd.DataFrame(entries)
        df_res.to_csv(output_folder + f"{d}_r2_ttest_results.csv")
        print(f"---- SAVED {d} t-test results")

if __name__ == "__main__":
    output_folder = "/Users/taichi/Desktop/master_thesis/results/v7/"
    t_test(output_folder)
