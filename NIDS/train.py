"""
This file contains code to train a Kitnet for conn.log data using the parameters from the best model.
Arguments: the dates of the data to be trained on.
#TODO: to be determined
Command: root@worker19:/opt/zeek/kitnet# python3 train_kitnet.py 2023-11-19

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
# The different files the AD can operate on
CONN_AD_ENABLED=False
HTTP_AD_ENABLED=False
DNS_AD_ENABLED=False
SSH_AD_ENABLED=False
SSL_AD_ENABLED=False

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
    global CONN_AD_ENABLED, DNS_AD_ENABLED, HTTP_AD_ENABLED, SSH_AD_ENABLED, SSL_AD_ENABLED
    parser = argparse.ArgumentParser(
        description='Trains a KitNET model on the specified log directory. The logs MUST have been stored in JSON format.')
    # Eventually we will need to implement some sort of directory to house these as people will retrain
    # and will still need access to historical models
    parser.add_argument('--model-path', type=str, default='kit.joblib',  
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
    parser.add_argument('--modules', nargs='+', required=True, choices=['CONN', 'DNS', 'HTTP'],
                        help='List of modules to enable. Choose from CONN, DNS, or HTTP. At least one module is required.')
    args = parser.parse_args()
    log_dir = args.log_dir
    # At least 1 module must be specified
    if 'CONN' in args.modules:
        CONN_AD_ENABLED = True
    if 'DNS' in args.modules:
        DNS_AD_ENABLED = True
    if 'HTTP' in args.modules:
        HTTP_AD_ENABLED = True
    if 'SSH' in args.modules:
        SSH_AD_ENABLED = True
    if 'SSL' in args.modules:
        SSL_AD_ENABLED = True
    
    # create kitnet model
    if CONN_AD_ENABLED:
        kit_conn_model = KitNet(
            max_size_ae=args.max_size_ae, 
            grace_feature_mapping=args.grace_feature_mapping, 
            grace_anomaly_detector=args.grace_anomaly_detector, 
            learning_rate=args.learning_rate, 
            hidden_ratio=args.hidden_ratio 
        )
    if DNS_AD_ENABLED:
        kit_dns_model = KitNet(
            max_size_ae=args.max_size_ae, 
            grace_feature_mapping=args.grace_feature_mapping, 
            grace_anomaly_detector=args.grace_anomaly_detector, 
            learning_rate=args.learning_rate, 
            hidden_ratio=args.hidden_ratio 
        )
    if HTTP_AD_ENABLED:
        kit_http_model = KitNet(
            max_size_ae=args.max_size_ae, 
            grace_feature_mapping=args.grace_feature_mapping, 
            grace_anomaly_detector=args.grace_anomaly_detector, 
            learning_rate=args.learning_rate, 
            hidden_ratio=args.hidden_ratio 
        )
    if SSH_AD_ENABLED:
        kit_ssh_model = KitNet(
            max_size_ae=args.max_size_ae, 
            grace_feature_mapping=args.grace_feature_mapping, 
            grace_anomaly_detector=args.grace_anomaly_detector, 
            learning_rate=args.learning_rate, 
            hidden_ratio=args.hidden_ratio 
        )
    if SSL_AD_ENABLED:
        kit_ssl_model = KitNet(
            max_size_ae=args.max_size_ae, 
            grace_feature_mapping=args.grace_feature_mapping, 
            grace_anomaly_detector=args.grace_anomaly_detector, 
            learning_rate=args.learning_rate, 
            hidden_ratio=args.hidden_ratio 
        )
    logging.info(f"Using Modules {args.modules}")
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
                if "conn." in file or "dns." in file or "http." in file:
                    # get the whole file in memory
                    if "conn." in file and CONN_AD_ENABLED:
                        logging.info(f"Opening file {current_file_path}")
                        json_data_file = ungzip(current_file_path) 
                        try:    
                            json.loads(json_data_file.split('\n')[0])
                        except json.JSONDecodeError as e:
                            logging.error(f"File {current_file_path} is not JSON. Skipping.")
                            continue 
                        np_arr = preprocess_json_conn(json_data_file)
                        train_batch(kit_conn_model, np_arr)
                    elif "dns." in file and DNS_AD_ENABLED:
                        logging.info(f"Opening file {current_file_path}")
                        json_data_file = ungzip(current_file_path) 
                        try:    
                            json.loads(json_data_file.split('\n')[0])
                        except json.JSONDecodeError as e:
                            logging.error(f"File {current_file_path} is not JSON. Skipping.")
                            continue 
                        np_arr = preprocess_json_dns(json_data_file)
                        train_batch(kit_dns_model, np_arr)  
                    elif "http." in file and HTTP_AD_ENABLED:
                        logging.info(f"Opening file {current_file_path}")
                        json_data_file = ungzip(current_file_path) 
                        try:    
                            json.loads(json_data_file.split('\n')[0])
                        except json.JSONDecodeError as e:
                            logging.error(f"File {current_file_path} is not JSON. Skipping.")
                            continue 
                        np_arr = preprocess_json_http(json_data_file)
                        train_batch(kit_http_model, np_arr)  
                    elif "ssh." in file and SSH_AD_ENABLED:
                        logging.info(f"Opening file {current_file_path}")
                        json_data_file = ungzip(current_file_path) 
                        try:    
                            json.loads(json_data_file.split('\n')[0])
                        except json.JSONDecodeError as e:
                            logging.error(f"File {current_file_path} is not JSON. Skipping.")
                            continue 
                        np_arr = preprocess_json_ssh(json_data_file)
                        train_batch(kit_ssh_model, np_arr)  
                    elif "ssl." in file and SSL_AD_ENABLED:
                        logging.info(f"Opening file {current_file_path}")
                        json_data_file = ungzip(current_file_path) 
                        try:    
                            json.loads(json_data_file.split('\n')[0])
                        except json.JSONDecodeError as e:
                            logging.error(f"File {current_file_path} is not JSON. Skipping.")
                            continue 
                        np_arr = preprocess_json_ssl(json_data_file)
                        train_batch(kit_ssl_model, np_arr)  

    # TODO: Before we exit the main function, dump the trained model to disk
    if CONN_AD_ENABLED:
        dump(kit_conn_model, "conn_" + args.model_path) 
        logging.info(f"Model is saved successfully as conn_{args.model_path}.") 
    if DNS_AD_ENABLED:
        dump(kit_dns_model, "dns_" + args.model_path) 
        logging.info(f"Model is saved successfully as dns_{args.model_path}.") 
    if HTTP_AD_ENABLED:
        dump(kit_http_model, "http_" + args.model_path) 
        logging.info(f"Model is saved successfully as http_{args.model_path}.") 
    if SSH_AD_ENABLED:
        dump(kit_ssh_model, "ssh_" + args.model_path) 
        logging.info(f"Model is saved successfully as ssh_{args.model_path}.") 
    if SSL_AD_ENABLED:
        dump(kit_ssl_model, "ssl_" + args.model_path) 
        logging.info(f"Model is saved successfully as ssl_{args.model_path}.") 

if __name__ == "__main__":
    main()