"""
This file contains code to train a Kitnet for HSRN data using the parameters from the best model.
Arguments: the dates of the data to be trained on.
#TODO: to be determined
Command: root@worker19:/opt/zeek/kitnet# python3 train_kitnet.py 2023-11-19

Author: Zoe Hsu 
"""
import argparse
import mlflow
import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pysad
from pysad.models.kitnet import KitNet
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import sys
import seaborn as sns
import logging
import os 
from data_preprocess import * 
from joblib import dump, load
import subprocess
import json 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s)')
MODEL_PATH = 'kit.joblib'
LOAD_MODEL = False        

def main(date):
    logging.info(msg="Started train_kitnet.py. ")
    # Load the latest data from the logs 
    preprocess_command = f"python3 data_preprocess.py {date}"
    subprocess.run(preprocess_command, shell=True)
    # Load the preprocessed data
    train = pd.read_csv(f"df_{date}.csv")  # Adjust the file name based on your output
    print("The data is loaded successfully! ")

    param_grid = {
    "max_size_ae": [30],#5-20 
    "grace_feature_mapping": [5000],#FMgrace = 5000 
    "grace_anomaly_detector": [50000], #ADgrace = 50000
    "learning_rate": [0.001] ,#0.001 to 0.1. #(Default=0.1).
    "hidden_ratio" : [0.5]#0.5 to 1.5 #(Default=0.75).
    }

    #Define the model 
    params = {
        param_name: random.choice(param_values)
        for param_name, param_values in param_grid.items()
    }
    logging.info(f"Run kitNET: using parameters {params}")
    with mlflow.start_run(run_name= 'Deploying KitNET') as run: 
        mlflow.log_param('model_name', "KitNET_wo_agg")
        # log the parameters used
        for key, val in params.items():
            mlflow.log_param(key, val)
        kit = KitNet(max_size_ae=params['max_size_ae'], grace_feature_mapping=params['grace_feature_mapping'], grace_anomaly_detector=params['grace_anomaly_detector'], learning_rate=params['learning_rate'], hidden_ratio=params['hidden_ratio'])
  
        # Loading a previous stored model from MODEL_PATH variable
        if LOAD_MODEL == True:
            kit = load(MODEL_PATH)
            print("Model is loaded successfully! ")

        logging.info("Training model")
        kit.fit(np.array(train)) #The output of KitNET is the RMSE anomaly score
        logging.info("Scoring model")

        #Explain the score distribution for this dataset
        score_train = kit.score(np.array(train))
        score_series = pd.Series(score_train)
        summary_stats = score_series.describe()

        # Log the summary statistics
        mlflow.log_metric("score_mean", summary_stats["mean"])
        mlflow.log_metric("score_std", summary_stats["std"])
        mlflow.log_metric("score_min", summary_stats["min"])
        mlflow.log_metric("score_max", summary_stats["max"])

        # Save the model
        dump(kit, 'kit.joblib')
        logging.info("Model is saved successfully as kit.joblib.")
        print("Model is saved successfully as kit.joblib. ")
        # Update the train record
        update_train_record(date)
        print("The train record is updated successfully! ")
        mlflow.end_run()

        

def update_train_record(date):

    record_file = 'train_record.json'
    # Check if the file exists
    if os.path.exists(record_file):
        # Load existing dates
        with open(record_file, 'r') as file:
            existing_dates = json.load(file)
    else:
        # If the file doesn't exist, create an empty list
        existing_dates = []

    # Append the current date to the list
    existing_dates.append(date)

    # Write the updated list back to the file
    with open(record_file, 'w') as file:
        json.dump(existing_dates, file, indent=2)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: train_kitnet.py. <date>")
        sys.exit(1)
    date = sys.argv[1]
    main(date)

    #remove the df file
    df_file = f"df_{date}.csv"
    #subcommand remove df_{date}.csv
    try:
        os.remove(df_file)
        print(f"File {df_file} removed successfully.")
    except FileNotFoundError:
        print(f"File {df_file} not found. It may have already been removed.")
    