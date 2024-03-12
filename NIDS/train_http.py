"""
This file contains code to train an AE-based model for http.log data using the parameters from the best model.
Arguments: the dates of the data to be trained on.
#TODO: change to AE instead of KitNet 

Command: python train_http.py --log-dir /usr/local/logs

Authors: 
- Zoe Hsu <wh2405@nyu.edu> 
- Olive Song <js10417@nyu.edu> 
- Diego Lopez <dtl310@nyu.edu>
- Zihang Xia <zx961@nyu.edu>
"""
import argparse
import random
import pandas as pd
import numpy as np
import pysad
from pysad.models.kitnet import KitNet
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import sys
import logging
import os 
#from NIDS.package.data_preprocess import * 
from joblib import dump, load
import subprocess
import json 
import gzip
from utils import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s)')
#MODEL_PATH = 'kit.joblib'
#LOAD_MODEL = False       

def ungzip(file_path):
    """
    Take a file path and ungzip it
    """
    # TODO: there should probably be some error checks here
    ungzipped_file_path = file_path.removesuffix('.gz')
    with gzip.open(file_path, 'rb') as gz_file:
        file_content = gz_file.read()
    return file_content.decode('utf-8')

def train_batch(kit, np_arr):
    """
    takes a 2D numpy array and trains the next batch for the KitNET model.
    Please use fit_partial() for this. Also, you will need to pass the model to this 
    function so please instantiate it somewhere. Keep it in memory and right before 
    main exits we dump it to disk.
    """

    assert len(np_arr.shape) <= 2, "The input array must be shape (n_samples, n_features) or (n_features,)."     
    kit.fit(np_arr) 

    return kit 

def main():
    """
    The main control flow for the application. 
    Taking the user-provided log directory as input, recursively searches the directory 
    for all date-based subdirectories, and trains models on all `conn.log` files. 
    """
    parser = argparse.ArgumentParser(
        description='Trains a KitNET model on the specified log directory. The logs MUST have been stored in JSON format.')
    # Eventually we will need to implement some sort of directory to house these as people will retrain
    # and will still need access to historical models
    parser.add_argument('--model-path', type=str, default='kit_http.joblib',  
                        help='The path to the model file to dump.') 
    parser.add_argument('--log-dir', type=str, required=True, 
                        help='Zeek logdir variable, where this script can find Zeek data.') 
    parser.add_argument('--max-size-ae', type=int, default=30, 
                        help='The maximum size of the autoencoder.')    
    parser.add_argument('--grace-feature-mapping', type=int, default=5000,  
                        help='The grace period for feature mapping.')     
    parser.add_argument('--grace-anomaly-detector', type=int, default=50000,    
                        help='The grace period for the anomaly detector.')  
    parser.add_argument('--learning-rate', type=float, default=0.001,    
                        help='The learning rate for the model.')
    parser.add_argument('--hidden-ratio', type=float, default=0.5,  
                        help='The hidden ratio for the model.')  
    args = parser.parse_args()
    log_dir = args.log_dir
    # create kitnet model
    kit = KitNet(
        max_size_ae=args.max_size_ae, 
        grace_feature_mapping=args.grace_feature_mapping, 
        grace_anomaly_detector=args.grace_anomaly_detector, 
        learning_rate=args.learning_rate, 
        hidden_ratio=args.hidden_ratio 
    )
    logging.info(f"Using logdir: {log_dir}") 
    logging.info(
        f"Using Parameters - max_size_ae: {args.max_size_ae}, "
        f"grace_feature_mapping: {args.grace_feature_mapping}, "
        f"grace_anomaly_detector: {args.grace_anomaly_detector}, "
        f"learning_rate: {args.learning_rate}, "
        f"hidden_ratio: {args.hidden_ratio}"
    )
    for sub_dir in os.listdir(log_dir):
        current_dir_path = os.path.join(log_dir, sub_dir) 
        # skip non-directory items in the top level logs/ folder
        if not os.path.isdir(current_dir_path): 
            continue     
        # `current` is a symlink for the current-day logs, we should not train on them as these files are in use. 
        if not os.path.islink(current_dir_path):
            # sub_dir is now any given historical data directory 
            logging.info(f"Checking {current_dir_path}")
            for file in os.listdir(current_dir_path):
                # file is now any given file in the historical data directory
                current_file_path = os.path.join(current_dir_path, file)
                if "http." in file:
                    # get the whole file in memory
                    logging.info(f"Opening file {current_file_path}")
                    json_data_file = ungzip(current_file_path) 
                    try:    
                        json.loads(json_data_file.split('\n')[0])
                    except json.JSONDecodeError as e:
                        logging.error(f"File {current_file_path} is not JSON. Skipping.")
                        continue 
                    np_arr = preprocess_json_http(json_data_file)
                    train_batch(kit, np_arr)

    # TODO: Before we exit the main function, dump the trained model to disk
    dump(kit, args.model_path) 
    logging.info(f"Model is saved successfully as {args.model_path}.") 



if __name__ == "__main__":
    main()