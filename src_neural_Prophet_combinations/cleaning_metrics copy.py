import os
import pandas as pd
import numpy as np

def calculate_metrics_by_model(base_folder):
    """
    Calcula as métricas de avaliação para cada combinação de modelo (covariáveis),
    excluindo datasets específicos da média geral.

    Resumo: Lê todos os arquivos metrics.csv nas subpastas, agrega os dados por modelo
    (combinação de covariáveis) e salva os resultados em um novo arquivo CSV.
    Exclui datasets específicos ao calcular a média geral.

    Variáveis de input:
    - base_folder (str): Caminho para a pasta base contendo as subpastas com os arquivos metrics.csv.

    Variáveis de saída:
    Nenhuma. Os resultados são salvos em um arquivo CSV.
    """
    all_metrics = []
    datasets_to_exclude = [
        "stretchedSociety_block_91_MAC001165",
        "establishedAffluence_block_25_MAC003004",
        "stretchedSociety_block_108_MAC005528",
        "establishedAffluence_block_28_MAC004904",
        "establishedAffluence_block_16_MAC002096",
        "stretchedSociety_block_97_MAC001314",
        "stretchedSociety_block_103_MAC004513",
        "stretchedSociety_block_105_MAC002128"
    ]

    # Percorre todas as subpastas
    for root, dirs, files in os.walk(base_folder):
        for file in files:
            if file == 'metrics.csv':
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)
                
                # Adiciona o nome da pasta (combinação de covariáveis) como uma coluna
                df['covariates'] = os.path.basename(root)
                all_metrics.append(df)

    # Combina todos os resultados
    final_metrics = pd.concat(all_metrics, ignore_index=True)

    # Filtra os datasets excluídos
    final_metrics_filtered = final_metrics[~final_metrics['dataset'].isin(datasets_to_exclude)]

    # Calcula a média das métricas para cada modelo (combinação de covariáveis)
    final_avg_metrics = final_metrics_filtered.groupby('model').agg({
        'mae': 'mean',
        'mape': 'mean',
        'smape': 'mean',
        'rmse': 'mean',
        'training_time': 'mean',
        'covariates': 'first'  # Mantém o nome da combinação de covariáveis
    }).reset_index()

    # Reordena as colunas
    final_avg_metrics = final_avg_metrics[['model', 'covariates', 'mae', 'mape', 'smape', 'rmse', 'training_time']]

    # Ordena os resultados pelo MAE (você pode mudar para outra métrica se preferir)
    final_avg_metrics = final_avg_metrics.sort_values('mae')

    # Salva os resultados em um novo arquivo CSV
    output_file = os.path.join(base_folder, 'metrics_by_model_combination_filtered.csv')
    final_avg_metrics.to_csv(output_file, index=False)
    print(f"Resultados salvos em: {output_file}")

    # Imprime as 5 melhores combinações baseadas no MAE
    print("\nTop 5 combinações de covariáveis baseadas no MAE (excluindo datasets específicos):")
    print(final_avg_metrics[['model', 'mae', 'mape', 'smape', 'rmse']].head())

    # Calcula e salva as estatísticas dos datasets excluídos separadamente
    excluded_metrics = final_metrics[final_metrics['dataset'].isin(datasets_to_exclude)]
    excluded_avg_metrics = excluded_metrics.groupby('model').agg({
        'mae': 'mean',
        'mape': 'mean',
        'smape': 'mean',
        'rmse': 'mean',
        'training_time': 'mean',
        'covariates': 'first'
    }).reset_index()

    excluded_output_file = os.path.join(base_folder, 'metrics_excluded_datasets.csv')
    excluded_avg_metrics.to_csv(excluded_output_file, index=False)
    print(f"\nMétricas dos datasets excluídos salvas em: {excluded_output_file}")

# Uso da função
base_folder = 'results/eval_metrics/'
calculate_metrics_by_model(base_folder)
