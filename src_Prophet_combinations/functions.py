import itertools
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import time
import pandas as pd
import numpy as np
import os
from omegaconf import OmegaConf

def load_config(config_path):
    return OmegaConf.load(config_path)

def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

def create_folders(base_folder, eval_folder, pred_folder, combo_name):
    combo_eval_folder = os.path.join(base_folder, eval_folder, combo_name)
    combo_pred_folder = os.path.join(base_folder, pred_folder, combo_name)
    
    os.makedirs(combo_eval_folder, exist_ok=True)
    os.makedirs(combo_pred_folder, exist_ok=True)
    
    return combo_eval_folder, combo_pred_folder

def train_prophet_model(df_train, combo):
    model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
    
    for cov in combo:
        model.add_regressor(cov)
    
    model.fit(df_train[['ds', 'y'] + list(combo)])
    
    return model

def make_predictions(model, df_test, combo):
    future = df_test[['ds'] + list(combo)]
    forecast = model.predict(future)
    return forecast['yhat'].values

def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    smape = symmetric_mean_absolute_percentage_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    return mae, mape, smape, rmse

def save_results(results, eval_folder, pred_folder, df_name, df_test, y_pred):
    # Save evaluation metrics (append if file exists)
    metrics_file = os.path.join(eval_folder, 'metrics.csv')
    if os.path.exists(metrics_file):
        existing_metrics = pd.read_csv(metrics_file)
        updated_metrics = pd.concat([existing_metrics, results], ignore_index=True)
        updated_metrics.to_csv(metrics_file, index=False)
    else:
        results.to_csv(metrics_file, index=False)
    
    # Save predictions
    df_test_pred = df_test.copy()
    df_test_pred['y_pred'] = y_pred
    df_test_pred.to_csv(os.path.join(pred_folder, f'{df_name}_predictions.csv'), index=False)

def prophet_with_covariates(df: pd.DataFrame, df_name: str, conf: OmegaConf) -> pd.DataFrame:
    df_name = df_name.split('.')[0]
    df_prophet = df.rename(columns={'time': 'ds', 'Energy_kwh': 'y'})
    
    covariates = ['holiday', 'month', 'hour', 'dayofweek_num', 'precipType', 'bool_weather_missing_values']
    covariate_combinations = list(itertools.combinations(covariates, 4))
    
    train_size = len(df) - 168
    df_train = df_prophet[:train_size]
    df_test = df_prophet[train_size:]
    
    results_list = []
    
    for combo in covariate_combinations:
        combo_name = '_'.join(sorted(combo))
        print(f"Running Prophet model with covariates: {combo_name} for dataset: {df_name}")
        
        eval_folder, pred_folder = create_folders(
            conf.models.results,
            conf.models.eval_metrics,
            conf.models.predictions,
            combo_name
        )
        
        start_time = time.time()
        model = train_prophet_model(df_train, combo)
        training_time = time.time() - start_time
        
        y_pred = make_predictions(model, df_test, combo)
        y_true = df_test['y'].values
        
        mae, mape, smape, rmse = calculate_metrics(y_true, y_pred)
        
        results = pd.DataFrame({
            'dataset': [df_name],
            'model': [f'Prophet_{combo_name}'],
            'mae': [mae],
            'mape': [mape],
            'smape': [smape],
            'rmse': [rmse],
            'training_time': [training_time]
        })
        
        save_results(results, eval_folder, pred_folder, df_name, df_test, y_pred)
        
        results_list.append(results)
    
    return pd.concat(results_list, ignore_index=True)

def save_all_results(all_results, conf):
    results_folder = conf.models.results
    all_results_file = os.path.join(results_folder, 'all_combinations_results.csv')
    if os.path.exists(all_results_file):
        existing_all_results = pd.read_csv(all_results_file)
        updated_all_results = pd.concat([existing_all_results, all_results], ignore_index=True)
        updated_all_results.to_csv(all_results_file, index=False)
    else:
        all_results.to_csv(all_results_file, index=False)



