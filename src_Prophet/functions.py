from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import time
import pandas as pd
import numpy as np


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))

# Função para calcular SMAPE
def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

def prophet(df:pd.DataFrame,df_name:str,eval_folder:str,pred_folder:str) -> None:

    df_name = df_name.split('.')[0]

    df_prophet = df.rename(columns={'time': 'ds', 'Energy_kwh': 'y'})


    train_size = len(df) - 168
    df_train = df_prophet[:train_size]
    df_test = df_prophet[train_size:]


    start_time = time.time()

    model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
    model.add_regressor('holiday')
    model.add_regressor('month')
    model.add_regressor('hour')
    model.add_regressor('dayofweek_num')

    model.fit(df_train)

    training_time = time.time() - start_time


    future = df_test[['ds', 'holiday', 'month', 'hour', 'dayofweek_num']]
    forecast = model.predict(future)


    y_true = df_test['y'].values
    y_pred = forecast['yhat'].values

    df_test['y_pred'] = y_pred

    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    smape = symmetric_mean_absolute_percentage_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))


    results = pd.DataFrame({
        'model': ['Prophet'],
        'mae': [mae],
        'mape': [mape],
        'smape': [smape],
        'rmse': [rmse],
        'training_time': [training_time]
    })

    results.to_csv(f'{eval_folder}{df_name}.csv', index=False)
    df_test.to_csv(f'{pred_folder}{df_name}.csv', index=False)
