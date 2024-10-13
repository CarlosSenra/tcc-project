import pandas as pd
import numpy as np
from neuralprophet import NeuralProphet
from sklearn.metrics import mean_absolute_error, mean_squared_error,mean_absolute_percentage_error
import time

# Função para calcular SMAPE
def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

def neural_prophet(df:pd.DataFrame,df_name:str,eval_folder:str,pred_folder:str) -> None:

    df_name = df_name.split('.')[0]

    print("Passo 1: Preparação dos dados")
    df_model = df[['time', 'Energy_kwh', 'holiday', 'month', 'hour', 'dayofweek_num']].copy()
    df_model = df_model.rename(columns={'time': 'ds', 'Energy_kwh': 'y'})

    print("\nPasso 2: Divisão dos dados")
    split_point = len(df_model) - 168
    df_train = df_model[:split_point]
    df_test = df_model[split_point:]
    print(f"Shape do conjunto de treino: {df_train.shape}")
    print(f"Shape do conjunto de teste: {df_test.shape}")

    print("\nPasso 3: Configuração do modelo Neural Prophet")
    model = NeuralProphet(
        daily_seasonality=True,
        weekly_seasonality=False,
        yearly_seasonality=False,
        batch_size=64,
        epochs=100,
        trainer_config = {"accelerator":"gpu"}
    )

    for regressor in ['holiday', 'month', 'hour', 'dayofweek_num']:
        model.add_future_regressor(regressor)

    print("\nPasso 4: Treinamento do modelo")
    start_time = time.time()
    metrics = model.fit(df_train, freq='H')
    training_time = time.time() - start_time

    print("\nPasso 5: Fazendo previsões")
    forecast = model.predict(df_test)


    print("\nPasso 6: Cálculo das métricas")
    y_true = df_test['y'].values
    y_pred = forecast['yhat1'].values

    df_test['y_pred'] = y_pred

    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    smape = symmetric_mean_absolute_percentage_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print("\nPasso 7: Salvando resultados")
    results = pd.DataFrame({
        'Model': ['NeuralProphet'],
        'MAE': [mae],
        'RMSE': [rmse],
        'MAPE': [mape],
        'SMAPE': [smape],
        'Training_Time': [training_time]
    })

    results.to_csv(f'{eval_folder}{df_name}.csv', index=False)
    df_test.to_csv(f'{pred_folder}{df_name}.csv', index=False)