import itertools
from neuralprophet import NeuralProphet
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import time
import pandas as pd
import numpy as np
import os
from omegaconf import OmegaConf

def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

def neural_prophet_with_covariates(df: pd.DataFrame, df_name: str, conf: OmegaConf) -> None:
    df_name = df_name.split('.')[0]
    
    # Passo 1: Preparação dos dados
    df_model = df[['time', 'Energy_kwh', 'holiday', 'month', 'hour', 'dayofweek_num', 'precipType', 'bool_weather_missing_values']].copy()
    df_model = df_model.rename(columns={'time': 'ds', 'Energy_kwh': 'y'})
    
    # Passo 2: Divisão dos dados
    split_point = len(df_model) - 168
    df_train = df_model[:split_point]
    df_test = df_model[split_point:]
    print(f"Shape do conjunto de treino: {df_train.shape}")
    print(f"Shape do conjunto de teste: {df_test.shape}")
    
    # Definindo as combinações de covariáveis
    covariates = ['holiday', 'month', 'hour', 'dayofweek_num', 'precipType', 'bool_weather_missing_values']
    covariate_combinations = list(itertools.combinations(covariates, 4))
    
    for combo in covariate_combinations:
        combo_name = '_'.join(sorted(combo))
        print(f"Running Neural Prophet model with covariates: {combo_name}")
        
        # Passo 3: Configuração do modelo Neural Prophet
        model = NeuralProphet(
            daily_seasonality=conf.neural_prophet.daily_seasonality,
            weekly_seasonality=conf.neural_prophet.weekly_seasonality,
            yearly_seasonality=conf.neural_prophet.yearly_seasonality,
            batch_size=conf.neural_prophet.batch_size,
            epochs=conf.neural_prophet.epochs,
            trainer_config={"accelerator": conf.neural_prophet.accelerator}
        )
        
        for regressor in combo:
            model.add_future_regressor(regressor)
        
        # Passo 4: Treinamento do modelo
        start_time = time.time()
        metrics = model.fit(df_train[['ds', 'y'] + list(combo)], freq='H')
        training_time = time.time() - start_time
        
        # Passo 5: Fazendo previsões
        forecast = model.predict(df_test[['ds'] + list(combo)])
        
        # Passo 6: Cálculo das métricas
        y_true = df_test['y'].values
        y_pred = forecast['yhat1'].values
        mae = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        smape = symmetric_mean_absolute_percentage_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # Passo 7: Salvando resultados
        results = pd.DataFrame({
            'Dataset': [df_name],
            'Model': [f'NeuralProphet_{combo_name}'],
            'MAE': [mae],
            'RMSE': [rmse],
            'MAPE': [mape],
            'SMAPE': [smape],
            'Training_Time': [training_time]
        })
        
        # Salvando em pastas específicas
        eval_folder = os.path.join(conf.models.results, conf.models.eval_metrics, combo_name)
        pred_folder = os.path.join(conf.models.results, conf.models.predictions, combo_name)
        os.makedirs(eval_folder, exist_ok=True)
        os.makedirs(pred_folder, exist_ok=True)
        
        results.to_csv(os.path.join(eval_folder, f'{df_name}_metrics.csv'), index=False)
        df_test_pred = df_test[['ds', 'y']].copy()
        df_test_pred['y_pred'] = y_pred
        df_test_pred.to_csv(os.path.join(pred_folder, f'{df_name}_predictions.csv'), index=False)
        
        # Salvando resultado individual na pasta parcial
        parcial_folder = os.path.join(conf.models.results, conf.models.parcial, combo_name)
        os.makedirs(parcial_folder, exist_ok=True)
        results.to_csv(os.path.join(parcial_folder, f'{df_name}_NeuralProphet.csv'), index=False)

def save_all_results(all_results, conf):
    results_folder = conf.models.results
    all_results_file = os.path.join(results_folder, 'all_results.csv')
    if os.path.exists(all_results_file):
        existing_all_results = pd.read_csv(all_results_file)
        updated_all_results = pd.concat([existing_all_results, all_results], ignore_index=True)
        updated_all_results.to_csv(all_results_file, index=False)
    else:
        all_results.to_csv(all_results_file, index=False)