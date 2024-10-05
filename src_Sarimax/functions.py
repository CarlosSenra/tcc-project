from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error,mean_absolute_percentage_error
import time
import pandas as pd
import numpy as np

def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

def sarimax(df:pd.DataFrame,df_name:str,eval_folder:str,pred_folder:str) -> None:

    df_name = df_name.split('.')[0]
    # Preparar os dados
    df_sarimax = df.set_index('time')

    train_size = len(df) - 168
    df_train = df_sarimax[:train_size]
    df_test = df_sarimax[train_size:]

    exog_vars = ['holiday', 'month', 'hour', 'dayofweek_num']

    start_time = time.time()

    # Aqui, estamos usando um modelo SARIMAX(1,1,1)(1,1,1,24) 
    model = SARIMAX(df_train['Energy_kwh'], 
                    exog=df_train[exog_vars],
                    order=(1, 1, 1),
                    seasonal_order=(1, 1, 1, 24),
                    enforce_stationarity=False,
                    enforce_invertibility=False)

    results = model.fit()

    training_time = time.time() - start_time

    forecast = results.get_forecast(steps=168, exog=df_test[exog_vars])
    y_pred = forecast.predicted_mean

    y_true = df_test['Energy_kwh'].values
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    smape = symmetric_mean_absolute_percentage_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    df_test['y_pred'] = y_pred
    df_test = df_test.reset_index()

    results = pd.DataFrame({
        'model': ['SARIMAX'],
        'mae': [mae],
        'mape': [mape],
        'smape': [smape],
        'rmse': [rmse],
        'training_time': [training_time]
    })

    results.to_csv(f'{eval_folder}{df_name}.csv', index=False)
    df_test.to_csv(f'{pred_folder}{df_name}.csv', index=False)
