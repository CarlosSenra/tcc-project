import os
from itertools import combinations
from omegaconf import OmegaConf

def get_config(yaml_name):
    conf = OmegaConf.load(yaml_name)
    return conf

def get_combinations_string(lista,combo):
    # Gera as combinações
    todas_combinacoes = []
    combinacoes_string = ["#"]
    for r in range(1, combo + 1):
        todas_combinacoes.extend(list(combinations(lista, r)))
    
    for elto in todas_combinacoes:
        combinacoes_string.append("-".join(elto))
    # Converte as tuplas em strings com underline
    return combinacoes_string

def mkdir_(destination_folder,name_folders):
    for name_dir in name_folders:
        try:
            os.makedirs(destination_folder+name_dir, exist_ok=True)
        except Exception as e:
            print(f"Erro ao criar diretório {name_dir}: {str(e)}")

def create_folders():
    mkdir_(config['paths']['eval_folder'],combinacoes)
    mkdir_(config['paths']['results_folder'],combinacoes)
    mkdir_(config['paths']['pred_folder'],combinacoes)


if __name__ == "__main__":
    lista = ['month', 'dayofweek_num', 'hour', 'holiday', 'bool_weather_missing_values', 'precipType']
    combinacoes = get_combinations_string(lista,1)

    config = get_config("lstm_config.yaml")

    create_folders()