from scipy.stats import ttest_ind, shapiro
import pandas as pd
import ast

def pair_exists(s_test_entries, d, m_1):
    return any(entry['Dimension'] == d and entry['Model'] == m_1 for entry in s_test_entries)

def t_test(output_folder, pairs):
    dimensions = ["MD", "CI", "FI", "IC", "P"]
    s_test_entries = []

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
            shapriso_res_m1 = shapiro(r2_m1)
            shapriso_res_m2 = shapiro(r2_m2)
            # print(f"shapiro : {m_1} : {shapriso_res_m1.statistic}")
            # print(f"shapiro : {m_2} : {shapriso_res_m2.statistic}")

            if not pair_exists(s_test_entries, d, m_1):
                s_test_entries.append({
                    "Dimension" : d,
                    "Model" : m_1,
                    "Shapiro-Wilk statistics" : shapriso_res_m1.statistic
                })

            if not pair_exists(s_test_entries, d, m_2):
                s_test_entries.append({
                    "Dimension" : d,
                    "Model" : m_2,
                    "Shapiro-Wilk statistics" : shapriso_res_m2.statistic
                })

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

    df_stest = pd.DataFrame(s_test_entries)
    df_stest.to_csv(output_folder + f"r2_normality_checks.csv")

if __name__ == "__main__":
    output_folder = "/Users/taichi/Desktop/master_thesis/results/v7/"
    t_test(output_folder)
