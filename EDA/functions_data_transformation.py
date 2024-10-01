import pandas as pd
import numpy as np
import random
import os
import json
from omegaconf import OmegaConf

conf = OmegaConf.load('../config.yaml')

def get_household(csv_file):
    def read_house_csv(csv_file):
        df = pd.read_csv(conf["EDA"]["folders"]["staging_data_folder"] + csv_file)
        return df 

    holidays = pd.read_csv(conf["EDA"]["dataframes"]["df_uk_holidays"])
    holidays['Bank holidays'] = pd.to_datetime(holidays['Bank holidays'])
    holidays = holidays.loc[(holidays['Bank holidays'].dt.year >= 2013)]
    df = read_house_csv(csv_file)
    df = df.rename(columns={'LCLid':'house_hold','tstp':'time','energy(kWh/hh)':'Energy_kwh'})
    df.iloc[df[df["Energy_kwh"].isin(["Null"])]["Energy_kwh"].index,3] = '0'
    df.Energy_kwh = pd.to_numeric(df.Energy_kwh)
    df.time = pd.to_datetime(df.time)
    df = df.loc[((df['time'].dt.year >= 2013) & (df.time.dt.month.isin([1,2,3,4,5,6,7,8,9,10,11,12]))) | (df['time'].dt.year == 2014)]
    df['holiday'] = df['time'].isin(holidays['Bank holidays']) #(df.time.dt.day_of_week >= 5) | (df['time'].isin(holidays['Bank holidays']))
    df['holiday'] = df['holiday'].astype(int)
    #df = df[df.house_hold == house_selected]
    
    return df

def transform_half_in_hourly(df : pd.DataFrame):
    """ 
        Description : a function where transform a dataframe with a halfhourly datetime in a hourly datetime dataframe
        Input : df : pd.Dataframe
        return : df : pd.Dataframe
    """
    list_df = []
    for house in df.house_hold.unique():
        df_temp = df[df['house_hold'] == house]
        df_temp = df[['time','Energy_kwh']]
        df_temp.set_index('time',inplace=True)
        df_temp = df_temp.resample('h').sum()
        df_temp['house_hold'] = house
        df_temp = df_temp.reset_index()
        list_df.append(df_temp)
    df_final = pd.concat(list_df, ignore_index=True)
    print(df_final.shape)
    return df_final


def add_weater_data(df:pd.DataFrame,df_weather:pd.DataFrame):
    df_merged = pd.merge(df,df_weather,how='left',on = 'time')
    print(df_merged.head())
    return df_merged

def add_holidays(df:pd.DataFrame):
    holidays = pd.read_csv('uk_bank_holidays.csv')
    holidays['Bank holidays'] = pd.to_datetime(holidays['Bank holidays'])
    holidays = holidays.loc[(holidays['Bank holidays'].dt.year >= 2013)]
    df['holiday'] = df['time'].isin(holidays['Bank holidays']) #(df.time.dt.day_of_week >= 5) | (df['time'].isin(holidays['Bank holidays']))
    df['holiday'] = df['holiday'].astype(int)
    return df

def add_bool_weather_missing_values(df:pd.DataFrame):
    df['bool_weather_missing_values'] = df['temperature'].apply(lambda x: 1 if x == np.nan else 0)
    return df

def dict_of_dict_labels(df:pd.DataFrame):
    """a function where get all categorical columns in a df and save ache categorie as a number in a dict, for the end all dicts is added in a final dict
    """
    dict_of_dict_labels = {}
    categorical_cols = df.select_dtypes(include='object').columns
    for column in categorical_cols:
        dict_label = {}
        catogories = df[column].unique()
        for i in range(len(catogories)):
            dict_label[catogories[i]] = i
        dict_of_dict_labels[column] = dict_label
    
    return dict_of_dict_labels

def labeling_categorical_features(df:pd.DataFrame):
    dict_labels = dict_of_dict_labels(df)
    
    for key,value in dict_labels.items(): #value is a dict
        label_dict = {k: v for k, v in value.items()}
        df[key] = df[key].map(label_dict).astype(str)
    
    return df, dict_labels


#if add a diferent feature in the future change this function
def feature_eng_function(df:pd.DataFrame):
    df['year'] = df['time'].dt.year.astype(str)
    df['month'] = df['time'].dt.month.astype(str)
    df['day'] = df['time'].dt.day.astype(str)
    df['hour'] = df['time'].dt.hour.astype(str)
    df['dayofweek_num'] = df['time'].dt.weekday.astype(str)
    df['temperature'] = df['temperature'].apply(lambda x: 0 if x == np.nan else x)
    df['windSpeed'] = df['windSpeed'].apply(lambda x: 0 if x == np.nan else x)
    df, dict_labels = labeling_categorical_features(df)
    return df, dict_labels


def save_json(dict_labels,house_selcted):
    file_path = f'{house_selcted}.json'  

    with open(file_path, "w") as f:
        json.dump(dict_labels, f)
        
    source_path = f'{house_selcted}.json'
    destination_path = f'json_files\{house_selcted}.json'

    # Check if source file exists
    if os.path.isfile(source_path):
        try:
            # Move the file using os.rename()
            os.rename(source_path, destination_path)
            print(f"File moved successfully from {source_path} to {destination_path}")
        except OSError as e:
            # Handle potential errors during move
            if "Destination exists" in str(e):  # Check for specific error
                print(f"Error: A file already exists at {destination_path}.")
            else:
                print(f"An error occurred during the move: {e}")
    else:
        print(f"Error: Source file '{source_path}' not found.")