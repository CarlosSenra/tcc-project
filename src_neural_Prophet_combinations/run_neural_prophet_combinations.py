import pandas as pd
import numpy as np
from neuralprophet import NeuralProphet
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import time
from itertools import combinations
import os
from omegaconf import OmegaConf

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

def prepare_data(df, config):
    time_col = config.data.time_column
    target_col = config.data.target_column
    covariates = config.data.covariates
    
    df_model = df[[time_col, target_col] + covariates].copy()
    df_model = df_model.rename(columns={time_col: 'ds', target_col: 'y'})
    split_point = len(df_model) - 168
    df_train = df_model[:split_point]
    df_test = df_model[split_point:]
    print(f"Shape of training set: {df_train.shape}")
    print(f"Shape of test set: {df_test.shape}")
    return df_train, df_test

def configure_model(config):
    model = NeuralProphet(
        daily_seasonality=config.neural_prophet.daily_seasonality,
        weekly_seasonality=config.neural_prophet.weekly_seasonality,
        yearly_seasonality=config.neural_prophet.yearly_seasonality,
        batch_size=config.neural_prophet.batch_size,
        epochs=config.neural_prophet.epochs,
        learning_rate=config.neural_prophet.learning_rate,
        trainer_config={"accelerator": config.neural_prophet.accelerator}
    )
    for regressor in config.data.covariates:
        model.add_future_regressor(regressor)
    return model

def train_model(model, df_train):
    start_time = time.time()
    metrics = model.fit(df_train, freq='H')
    training_time = time.time() - start_time
    return training_time

def make_predictions(model, df_test):
    forecast = model.predict(df_test)
    y_true = df_test['y'].values
    y_pred = forecast['yhat1'].values
    df_test['y_pred'] = y_pred
    return df_test, y_true, y_pred

def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    smape = symmetric_mean_absolute_percentage_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, mape, smape, rmse

def save_results(df_name, covariates, mae, rmse, mape, smape, training_time, eval_folder):
    results = pd.DataFrame({
        'Model': ['NeuralProphet'],
        'Dataset': [df_name],
        'Covariates': [', '.join(covariates)],
        'MAE': [mae],
        'RMSE': [rmse],
        'MAPE': [mape],
        'SMAPE': [smape],
        'Training_Time': [training_time]
    })
    
    metrics_file = os.path.join(eval_folder, 'metrics.csv')
    if os.path.exists(metrics_file):
        existing_results = pd.read_csv(metrics_file)
        updated_results = pd.concat([existing_results, results], ignore_index=True)
        updated_results.to_csv(metrics_file, index=False)
    else:
        results.to_csv(metrics_file, index=False)

def save_predictions(df_test, df_name, pred_folder):
    df_test.to_csv(os.path.join(pred_folder, f'{df_name}_predictions.csv'), index=False)

def neural_prophet_with_covariates(df: pd.DataFrame, df_name: str, config: OmegaConf, covariates: list) -> None:
    df_name = df_name.split('.')[0]
    
    folder_name = "_".join(covariates)
    eval_folder = os.path.join(config.models.eval_metrics, folder_name)
    pred_folder = os.path.join(config.models.predictions, folder_name)
    create_folder(eval_folder)
    create_folder(pred_folder)
    
    df_train, df_test = prepare_data(df, config)
    model = configure_model(config)
    training_time = train_model(model, df_train)
    df_test, y_true, y_pred = make_predictions(model, df_test)
    mae, mape, smape, rmse = calculate_metrics(y_true, y_pred)
    
    save_results(df_name, covariates, mae, rmse, mape, smape, training_time, eval_folder)
    save_predictions(df_test, df_name, pred_folder)

def run_neural_prophet_combinations(df: pd.DataFrame, df_name: str, config: OmegaConf):
    all_covariates = config.data.covariates
    covariate_combinations = list(combinations(all_covariates, 4))
    
    for cov_combination in covariate_combinations:
        print(f"\nRunning model with covariates: {cov_combination}")
        neural_prophet_with_covariates(df, df_name, config, list(cov_combination))

def main():
    config = OmegaConf.load('../config.yaml')
    csv_files = [f for f in os.listdir(config.models.train.dataframes_folder) if f.endswith('.csv')]

    for csv_file in csv_files:
        print(f"\nProcessing dataset: {csv_file}")
        df = pd.read_csv(os.path.join(config.models.train.dataframes_folder, csv_file))
        run_neural_prophet_combinations(df, csv_file, config)    


if __name__ == "__main__":

    main()