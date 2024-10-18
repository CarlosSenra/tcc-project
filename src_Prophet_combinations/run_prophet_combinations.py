from omegaconf import OmegaConf
import os
import pandas as pd
import functions

conf = OmegaConf.load('../config.yaml')

if __name__ == "__main__":

    df_folder = conf.models.train.dataframes_folder
    print(f"Loading datasets from: {df_folder}")

    all_results = pd.DataFrame()

    for dataset in os.listdir(df_folder):
        if dataset.endswith('.csv'):
            df_path = os.path.join(df_folder, dataset)
            df = pd.read_csv(df_path)
            results = functions.prophet_with_covariates(df, dataset, conf)
            all_results = pd.concat([all_results, results], ignore_index=True)

    functions.save_all_results(all_results, conf)