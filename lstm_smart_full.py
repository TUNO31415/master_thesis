from lstm_smart import lstm_smart_n_times_k_fold
import pandas as pd
from utils import data_loader
import os

paco_path = "/tudelft.net/staff-umbrella/tunoMSc2023/paco_dataset/"

def main():
    dimensions = ["MD", "CI", "FI", "IC", "P"]
    output_folder = paco_path + "result_lstm_full_res/"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    r2_results = []
    mse_results = []
    for d in dimensions:
        print(f"---- {d} DIMENSION START ----")
        X, Y = data_loader(d)
        res = lstm_smart_n_times_k_fold(X, Y)
        print("smart DONE")

        
        r2 = [a[0] for a in res]
        mse = [a[1] for a in res]
        r2_results.append(r2)
        mse_results.append(mse)
    
    df = pd.DataFrame({
        "Dimension" : dimensions,
        "r2 results" : r2_results,
        "mse results" : mse_results
    })

    df.to_csv(output_folder + "lstm_smart_full_results.csv", index=False)

if __name__ == "__main__":
    main()