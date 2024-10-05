from omegaconf import OmegaConf
import os
import pandas as pd
import functions

conf = OmegaConf.load('../config.yaml')


if __name__ == "__main__":

    df_folder = conf['models']['train']['dataframes_folder']
    print(df_folder)
    eval_folder = conf['models']['eval_metrics']
    pred_folder = conf['models']['predictions']

    for file_name in os.listdir(df_folder):
        print(file_name)
        df = pd.read_csv(f"{df_folder}{file_name}")
        df['time'] = pd.to_datetime(df['time'])
        df = df[['time','Energy_kwh','holiday','month','hour','dayofweek_num']]

        functions.neural_prophet(df = df,
                                 df_name = file_name,
                                 eval_folder = eval_folder,
                                 pred_folder = pred_folder)

        print('ok')