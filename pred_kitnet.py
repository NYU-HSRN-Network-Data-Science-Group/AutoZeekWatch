"""
This script takes in the data from "current" folder, that stores new data and preprocesses it for the kitnet model.

By Zoe

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
import pyod 
from joblib import dump, load
import json 
import tailer
from datetime import datetime


def main(new_instance):

    #Perform preprocessing on the new instance as in the train set
    # Load new data (latest instance of data)
    #df_merged = duration_to_numerical(new_instance)

    df = fill_na(new_instance)
    record = {'ts': df['ts'][0], 'uid': df['uid'][0], 'id.orig_h': df['id.orig_h'][0], 'id.orig_p': df['id.orig_p'][0], 'id.resp_h': df['id.resp_h'][0], 'id.resp_p': df['id.resp_p'][0], 'anomaly_score': 0}
    print("The record is :", record)
    df_preprocessed = preprocess(df)
    # make sure the columns are the same as the original df
    df_final = makedf_samecol(df_preprocessed)
    df_final = df_final.drop(columns=['orig_l2_addr','resp_l2_addr'])
    print("The dataframe is processed successfully, with the shape of ", df_final.shape)

    #Load existing model 
    kit = load('kit.joblib')

    # Make prediction 
    anomaly_score = kit.score(np.array(df_final))
    
    # Input : id.orig_p,id.resp_p,duration,orig_bytes,resp_bytes,missed_bytes,orig_pkts,orig_ip_bytes,resp_pkts,resp_ip_bytes,history_has_S,history_has_h,history_has_A,history_has_D,history_has_a,history_has_d,history_has_F,history_has_f,is_destination_broadcast,conn_state_OTH,conn_state_RSTO,conn_state_RSTRH,conn_state_S0,conn_state_S1,conn_state_SF,proto_tcp,proto_udp,traffic_direction_internal,traffic_direction_outgoing,service_dns,service_http,service_ntp,service_other,service_ssl,conn_state_REJ,conn_state_RSTOS0,conn_state_RSTR,conn_state_S2,conn_state_S3,conn_state_SH,conn_state_SHR,service_dhcp,service_irc,service_ssh,traffic_direction_external,traffic_direction_incoming
    #  Save prediction to json file (anomaly score, id.orig_p, id.resp_p)
    # Extract id.orig_p and id.resp_p from df_final
    if anomaly_score:
        record['anomaly_score'] = float(anomaly_score[0])

    update_record(record)


def update_record(dic_record):
    try:
        # Add the current date to the record
        current_date = datetime.now().strftime("%Y-%m-%d")
        record_file = f'/opt/zeek/anomaly_record/anomaly_record_{current_date}.txt'
        #record_file = f'../anomaly_record/anomaly_record_{current_date}.txt' #remember to create the folder first

        # Convert every value to string (excluding 'anomaly_score')
        for key, value in dic_record.items():
            if key != 'anomaly_score':
                dic_record[key] = str(value)

        # Write to the file
        with open(record_file, 'a') as convert_file: 
            # Move the cursor to a new line
            convert_file.write('\n')
            convert_file.write(json.dumps(dic_record))
        print(f"Record updated successfully in {record_file}")
    except Exception as e:
        print(f"Error updating record: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    # Follow the file as it grows

    # To an absolute path like this
    #log_file_path = '/opt/zeek/logs/current/conn.log'
    # for line in tailer.follow(open(log_file_path)):
    #     # Fix boolean values in JSON string
    #     line = line.replace('true', 'True').replace('false', 'False')
    #     # Replace single quotes with double quotes
    #     line = line.replace("'", "\"")
    #     # Create a DataFrame from the modified JSON string
    #     conn_dict = json.loads(line)
    #     df = pd.DataFrame([conn_dict], index=[0])
    #     try:
    #         main(df)
    #     except:
    #         print("There are some missing columns, so the fit_score won't work. \n")
    #         print("Here are the column names for this instance:", df.columns)


    # Alternative: Read the entire file into a Pandas DataFrame
    with open('../logs/current/conn.log') as file:
    #with open(log_file_path) as file:
        for line in file:
            if line.strip() :
                # Replace single quotes with double quotes
                line_with_double_quotes = line.replace("'", "\"")
                # Use json.loads
                conn_dict = json.loads(line_with_double_quotes)
                df = pd.DataFrame([conn_dict], index=[0])
                try:
                    main(df)
                except:
                    print("There are some missing columns, so the fit_score won't work. \n")
                    print("Here are the column names for this instance:", df.columns)

