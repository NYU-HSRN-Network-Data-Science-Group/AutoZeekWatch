"""
This script is used to preprocess the data for the 'KitNET' model.
Example command:
NIDS% python train.py --log-dir /usr/local/logs

There should be 46 columns in the new dataframe. 
For the online-normalization, we can skip it for now, since KitNet has its own normalization.

By Zoe Hsu

"""

import sys
import os
import subprocess
import pandas as pd
import gzip
import numpy as np
import json
import logging
import ipaddress
from scipy.stats import entropy

# TODO: is there a better way to handle multi-file logging aside from spamming these everywhere?
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s)')

def ungzip(file_path):
    """
    Take a file path and ungzip it
    """
    # TODO: there should probably be some error checks here

    # if the file is not a .gz file, read the content directly and return it 
    if (not file_path.endswith('.gz')) and (file_path.endswith('.log')):
        with open(file_path, 'rb') as f:
            return f.read().decode('utf-8') 
        
    ungzipped_file_path = file_path.removesuffix('.gz')
    with gzip.open(file_path, 'rb') as gz_file:
        file_content = gz_file.read()
    return file_content.decode('utf-8')


def preprocess_json_conn(json_batch):
    """
    This function receives a json batch from the main control flow of the train 
    functions. It should convert the conn.log of the json_batch to a numpy 2D array, apply necessary transformations,
    then return it. 

    Note: the input is only one unzipped json file. 
    """
    features = ["id.orig_h", "id.resp_h", "proto", "service", "duration", "conn_state", 
                "local_orig","local_resp","missed_bytes","history", 
                "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes"]
    #TODO: add features: duration, local_orig, local_resp 
    data_list = []
    for line in json_batch.splitlines():
        # log_entry is now a single json log from the file 
        log_entry = json.loads(line.strip())
        # data_list.append([log_entry[feature] for feature in features])
        # Check if each feature is present in the log_entry
        feature_values = [log_entry.get(feature, None) for feature in features]
        data_list.append(feature_values)
    #Re-use the preprocess function from last sem by Zoe. 
    #TODO: optimize the code via removing pandas
    new_df = pd.DataFrame(data_list, columns=features) 
    #Fill NaNs with 0s : duration, orig_bytes resp_bytes, if there are no columns, create one and fill with 0s 
    new_df = fill_na(new_df) 

    # create history, broadcast, traffic_direction variables
    new_df = create_history_variable(new_df)
    new_df = create_broadcast_variable(new_df)
    new_df = create_direction_variable(new_df)
    # one hot encode categorical variables
    column_name = ['conn_state', "proto", "traffic_direction" , "service"]
    new_df = one_hot_encode(new_df, column_name)
    # Convert the boolean values in columns "local_orig" and "local_resp" to 1 and 0s
    new_df['local_orig'] = new_df['local_orig'].astype(int)
    new_df['local_resp'] = new_df['local_resp'].astype(int)
    # make sure the columns are the same as the original df
    #TODO: to be confirmed once HSRN EDA is done
    cols = ['conn_state_OTH', 'conn_state_REJ','conn_state_RSTO', 'conn_state_RSTOS0', 'conn_state_RSTR','conn_state_RSTRH', 
        'conn_state_S0', 'conn_state_S1', 'conn_state_S2','conn_state_S3', 'conn_state_SF', 'conn_state_SH', 'conn_state_SHR',
        'proto_tcp', 'proto_udp', 
        'service_dhcp', 'service_dns','service_http', 'service_irc','service_ntp',
        'service_other', 'service_ssh','service_ssl',
        'traffic_direction_external','traffic_direction_incoming', 
        'traffic_direction_internal','traffic_direction_outgoing',
        "duration","local_orig","local_resp","missed_bytes","orig_pkts","orig_ip_bytes","resp_pkts","resp_ip_bytes"]
    new_df = makedf_samecol(cols, new_df)
    # Convert DataFrame to NumPy array
    np_arr = new_df.to_numpy(dtype=np.float32)
    return np_arr

from columns import Aggr_conn
def preprocess_json_conn_agg(json_batch):
    """
    This function receives a json batch from the main control flow of the train 
    functions. It should convert the conn.log of the json_batch to a numpy 2D array, apply necessary transformations,
    then return it. 

    Note: the input is only one unzipped json file. 
    """
    features = ["ts","uid", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p",
            "proto", "service", "duration", "conn_state", "local_orig","local_resp",
            "missed_bytes","history", "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes"]
    #TODO: add features: duration, local_orig, local_resp 
    data_list = []
    for line in json_batch.splitlines():
        # log_entry is now a single json log from the file 
        log_entry = json.loads(line.strip())
        # data_list.append([log_entry[feature] for feature in features])
        # Check if each feature is present in the log_entry
        feature_values = [log_entry.get(feature, None) for feature in features]
        data_list.append(feature_values)

    #TODO: optimize the code via removing pandas
    df = pd.DataFrame(data_list, columns=features) 

    #fill Nans with 0s : duration, orig_bytes resp_bytes
    df = fill_na(df)  
    # create history, broadcast, traffic_direction variables
    df = create_history_variable(df)
    df = create_broadcast_variable(df)
    df = create_direction_variable(df)

    # one hot encode categorical variables
    column_name = ['conn_state', "proto", "traffic_direction" , "service"]
    df = one_hot_encode(df, column_name)

    # Convert the boolean values in columns "local_orig" and "local_resp" to 1 and 0s
    df['local_orig'] = df['local_orig'].astype(int)
    df['local_resp'] = df['local_resp'].astype(int)

    #Compute Aggregated Features 
    windows = [60,3600,7200] #seconds 
    grp = ['id.orig_h', 'id.orig_p', 'id.resp_h', 'id.resp_p']
    aggr_feature_num = ['duration', 'missed_bytes', 'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes']
    aggr_feature_cat = ['local_orig', 'local_resp']
    for window in windows:
        for feature in aggr_feature_num:
            df = calculate_agg_feature_num(df, feature, window)
        for feature in aggr_feature_cat:
            df = calculate_agg_feature_cat(df, feature, window)
    cols = Aggr_conn
    # make sure the columns are the same 
    df = makedf_samecol(cols, df)        
    # Convert DataFrame to NumPy array
    np_arr = df.to_numpy(dtype=np.float32)
    return np_arr

def preprocess_json_dns(json_batch):
    """
    This function receives a json batch from the main control flow of the train 
    functions. It should convert the dns.log of the json_batch to a numpy 2D array, apply necessary transformations,
    then return it. 

    Note: the input is only one unzipped json file. 
    """
    features = ['id.orig_h', "id.resp_h", "proto", "rtt","qclass_name", "qtype_name","rcode_name",
                "AA","TC","RD","RA", "rejected"]
    data_list = []
    for line in json_batch.splitlines():
        # log_entry is now a single json log from the file 
        log_entry = json.loads(line.strip())
        # Check if each feature is present in the log_entry
        feature_values = [log_entry.get(feature, None) for feature in features]
        data_list.append(feature_values)
    df = pd.DataFrame(data_list, columns=features) 

    has_null = ['rtt', 'qclass_name', 'qtype_name', 'rcode_name']
    # Create a variable to track if the feature contains null. Create a column "has_null_featurename"
    for feature in has_null: 
        df[f'has_{feature}'] = df[feature].notnull().astype(int)
    
    # create broadcast, traffic_direction variables
    df = create_broadcast_variable(df)
    df = create_direction_variable(df)

    # one hot encode categorical variables: proto, qtype, qclass, rcode_name
    column_name = ['proto','qtype_name','qclass_name','rcode_name','traffic_direction']
    df = one_hot_encode(df, column_name)
    
    #encode boolean features 
    boolean_to_convert = ['AA', 'TC', 'RD', 'RA', 'rejected']
    df[boolean_to_convert] = df[boolean_to_convert].astype(int)

    #fillna with 0s:rtt
    columns_to_fill_with_zeros = ['rtt']
    df[columns_to_fill_with_zeros] = df[columns_to_fill_with_zeros].fillna(0)

    #same columns 
    #TODO: to be confirmed once HSRN EDA is done
    dns_cols = ['rtt', 'AA', 'TC', 'RD', 'RA', 'rejected',
       'has_rtt', 'has_qclass_name', 'has_qtype_name', 'has_rcode_name',
       'is_destination_broadcast', 
       'proto_tcp', 'proto_udp',
       'qtype_name_*', 'qtype_name_A',
       'qtype_name_AAAA', 'qtype_name_HTTPS', 'qtype_name_PTR',
       'qclass_name_C_INTERNET', 'qclass_name_qclass-32769',
       'rcode_name_NOERROR', 'rcode_name_NXDOMAIN', 
       'traffic_direction_IPv6',
       'traffic_direction_external','traffic_direction_incoming', 
        'traffic_direction_internal','traffic_direction_outgoing']

    df = makedf_samecol(dns_cols, df)

    # Convert DataFrame to NumPy array
    np_arr = df.to_numpy(dtype=np.float32)
    return np_arr

def preprocess_json_http(json_batch):
    """
    This function receives a json batch from the main control flow of the train 
    functions. It should convert the dns.log of the json_batch to a numpy 2D array, apply necessary transformations,
    then return it. 

    Note: the input is only one unzipped json file. 
    """
    features = ['id.orig_h', 'id.resp_h','trans_depth','method','host','version',
                'request_body_len','response_body_len','status_code']
        
    data_list = []
    for line in json_batch.splitlines():
        # log_entry is now a single json log from the file 
        log_entry = json.loads(line.strip())
        # Check if each feature is present in the log_entry
        feature_values = [log_entry.get(feature, None) for feature in features]
        data_list.append(feature_values)

    df = pd.DataFrame(data_list, columns=features) 

    has_null = ['host']
    # Create a variable to track if the feature contains null. Create a column "has_null_featurename"
    for feature in has_null: 
        df[f'has_{feature}'] = df[feature].notnull().astype(int)
    
    # create broadcast, traffic_direction variables
    df = create_broadcast_variable(df)
    df = create_direction_variable(df)

    # one hot encode categorical variables: proto, qtype, qclass, rcode_name
    column_name = ['version','method','status_code','traffic_direction']
    df = one_hot_encode(df, column_name)

    #TODO: to be confirmed once EDA is done
    http_cols = ['trans_depth', 'request_body_len',
       'response_body_len', 'has_host', 'is_destination_broadcast',
       'method_CONNECT', 'method_GET', 
       'status_code_0', 'status_code_200',
       'version_0.9', 'version_1.1',
       'traffic_direction_IPv6',
       'traffic_direction_external','traffic_direction_incoming', 
        'traffic_direction_internal','traffic_direction_outgoing']
    
    df = makedf_samecol(http_cols, df)

    # Convert DataFrame to NumPy array
    np_arr = df.to_numpy(dtype=np.float32)
    return np_arr


def preprocess_json_ssh(json_batch):
    """
    This function receives a json batch from the main control flow of the train 
    functions. It should convert the dns.log of the json_batch to a numpy 2D array, apply necessary transformations,
    then return it. 

    Note: the input is only one unzipped json file. 
    """
    features = ['id.orig_h', 'id.resp_h','trans_depth','method','host','version',
                'request_body_len','response_body_len','status_code']
        
    data_list = []
    for line in json_batch.splitlines():
        # log_entry is now a single json log from the file 
        log_entry = json.loads(line.strip())
        # Check if each feature is present in the log_entry
        feature_values = [log_entry.get(feature, None) for feature in features]
        data_list.append(feature_values)

    df = pd.DataFrame(data_list, columns=features) 

    # create broadcast, traffic_direction variables
    df = create_broadcast_variable(df)
    df = create_direction_variable(df)

    #label encode: auth_success, direction
    #TODO: ask Diego is it's the same as 'traffic direction'
    df['auth_success'] = df['auth_success'].replace({False: 0, True: 1})
    df['direction'] = df['direction'].replace({'OUTBOUND': 1, 'INBOUND': 0})

    # one hot encode categorical variables: proto, qtype, qclass, rcode_name
    column_name = ['version','traffic_direction']
    df = one_hot_encode(df, column_name)

    #TODO: to be confirmed once EDA is done
    ssh_cols = ['auth_success', 'auth_attempts', 'direction',
       'is_destination_broadcast', 'version_2', 
       'traffic_direction_external','traffic_direction_incoming', 
        'traffic_direction_internal','traffic_direction_outgoing']
    
    df = makedf_samecol(ssh_cols, df)

    # Convert DataFrame to NumPy array
    np_arr = df.to_numpy(dtype=np.float32)
    return np_arr

def preprocess_json_ssl(json_batch):
    """
    This function receives a json batch from the main control flow of the train 
    functions. It should convert the dns.log of the json_batch to a numpy 2D array, apply necessary transformations,
    then return it. 

    Note: the input is only one unzipped json file. 
    """
    features = ['id.orig_h', 'id.resp_h','version','resumed','established',
            'ssl_history','cert_chain_fps','client_cert_chain_fps','sni_matches_cert','validation_status']
    #Ignore 'cipher','curve','server_name','next_protocol' columns for now, we can include them if they are useful later on.

    data_list = []
    for line in json_batch.splitlines():
        # log_entry is now a single json log from the file 
        log_entry = json.loads(line.strip())
        # Check if each feature is present in the log_entry
        feature_values = [log_entry.get(feature, None) for feature in features]
        data_list.append(feature_values)

    df = pd.DataFrame(data_list, columns=features) 

    has_null = ['version', 'cert_chain_fps', 'client_cert_chain_fps', 'sni_matches_cert', 'validation_status']
    # Create a variable to track if the feature contains null. Create a column "has_null_featurename"
    for feature in has_null: 
        df[f'has_{feature}'] = df[feature].notnull().astype(int)
    

    # create broadcast, traffic_direction variables
    df = create_broadcast_variable(df)
    df = create_direction_variable(df)

    #fillna, considering null as False
    df['sni_matches_cert'] = df['sni_matches_cert'].fillna(False)
    df['cert_chain_fps'] = df['cert_chain_fps'].fillna("").apply(list)
    df['client_cert_chain_fps'] = df['client_cert_chain_fps'].fillna("").apply(list)

    #boolean to int
    boolean_to_convert = ['resumed','established','sni_matches_cert']
    df[boolean_to_convert] = df[boolean_to_convert].astype(int)
    # one hot encode categorical variables: version, traffic_direction
    column_name = ['version','traffic_direction']
    df = one_hot_encode(df, column_name)

    #Make the length of the cert_chain_fps and client_cert_chain_fps as a new feature
    df['cert_chain_fps'] = df['cert_chain_fps'].apply(lambda x: len(x))
    df['client_cert_chain_fps'] = df['client_cert_chain_fps'].apply(lambda x: len(x))

    #create history variable
    df = create_sslhistory_variable(df)

    #TODO: to be confirmed once EDA is done
    # List of new columns created from ssl_history
    # Define all possible characters in 'ssl_history'
    all_characters = set('HCSVTXKRNYGFWUABDOPMIJLQhcsvtxkrnygfwuabdopmijlq')
    history_col = [f'ssl_history_has_{char}' for char in all_characters]
    ssl_cols = ['resumed', 'established', 
        'cert_chain_fps', 'client_cert_chain_fps', 'sni_matches_cert',
        'has_version', 'has_cert_chain_fps',
        'has_client_cert_chain_fps', 'has_sni_matches_cert',
        'has_validation_status', 'is_destination_broadcast', 
        'version_TLSv12','version_TLSv13', 
        'traffic_direction_external','traffic_direction_incoming', 
        'traffic_direction_internal','traffic_direction_outgoing']
    ssl_cols = ssl_cols + history_col

    df = makedf_samecol(ssl_cols, df)

    # Convert DataFrame to NumPy array
    np_arr = df.to_numpy(dtype=np.float32)
    return np_arr

#==================================================================================
# def is_private_ip(ip_str):
#     """
#     Takes an IP string and returns whether the IP is private or not per RFC 1918.

#     Parameters
#     ----------
#     ip_str: str
#         String of an IP address.

#     Returns
#     -------
#     bool: a bool of whether or not the IP is private. 
#     """
#     octets = [int(x) for x in ip_str.split(".")]
#     if octets[0] == 10 \
#             or (octets[0] == 172 and 16 <= octets[1] <= 31) \
#             or (octets[0] == 192 and octets[1] == 168):
#         return True
#     else:
#         return False

def is_private_ip(ip_str):
    """
    Takes an IP string and returns whether the IP is private or not per RFC 1918.

    Parameters
    ----------
    ip_str: str
        String of an IP address.

    Returns
    -------
    bool: a bool of whether or not the IP is private.
    """
    try:
        ip = ipaddress.ip_address(ip_str)
        if ip.version == 4:
            return ip.is_private
        else:
            return False  # Ignore IPv6 addresses
    except ValueError:
        return False  # Invalid IP address format


def get_traffic_direction(source_ip, destination_ip):
    """
    Takes a source and destination IP address and returns the direction of the traffic.
    Please ensure the source and destination are correct as this is useless without the verification of the parameters.

    Parameters
    ----------
    source_ip: str
        Source IP address of the flow.
    destination_ip: str
        Destination IP address of the flow.
    
    Returns
    -------
    str: string indicating the direction. Can be 'internal', 'outgoing', 'incoming' or 'external'.
    """
    src_ip = ipaddress.ip_address(source_ip) 
    dest_ip = ipaddress.ip_address(destination_ip) 
    if src_ip.version == 6 or dest_ip.version ==6:
        return "IPv6"
    if is_private_ip(source_ip) and is_private_ip(destination_ip):
        return "internal"
    elif is_private_ip(source_ip) and not is_private_ip(destination_ip):
        return "outgoing"
    elif not is_private_ip(source_ip) and is_private_ip(destination_ip):
        return "incoming"
    else:
        return "external"

def drop_columns(new_df, columns_to_drop):
    columns_to_drop_existing = [col for col in columns_to_drop if col in new_df.columns]
    new_df.drop(columns=columns_to_drop_existing, axis=1, inplace=True)
    return new_df

def create_history_variable(new_df):
    # break out history variable
    
    if 'history' not in new_df.columns: 
        new_df['history'] = 'N'  

    #fill NaNs with 'N'
    new_df['history'] = new_df['history'].fillna('N') 
    new_df['history_has_S'] = new_df['history'].apply(lambda x: 1 if "S" in x else 0)
    new_df['history_has_h'] = new_df['history'].apply(lambda x: 1 if "h" in x else 0)
    new_df['history_has_A'] = new_df['history'].apply(lambda x: 1 if "A" in x else 0)
    new_df['history_has_D'] = new_df['history'].apply(lambda x: 1 if "D" in x else 0)
    new_df['history_has_a'] = new_df['history'].apply(lambda x: 1 if "a" in x else 0)
    new_df['history_has_d'] = new_df['history'].apply(lambda x: 1 if "d" in x else 0)
    new_df['history_has_F'] = new_df['history'].apply(lambda x: 1 if "F" in x else 0)
    new_df['history_has_f'] = new_df['history'].apply(lambda x: 1 if "f" in x else 0)
    new_df['history_has_N'] = new_df['history'].apply(lambda x: 1 if "N" in x else 0)
    new_df = new_df.drop(columns='history')
    return new_df 

def create_sslhistory_variable(new_df):
    # break out history variable
    
    if 'ssl_history' not in new_df.columns: 
        new_df['ssl_history'] = ''  #TODO:?

    #fill NaNs with 'N'
    new_df['ssl_history'] = new_df['ssl_history'].fillna('') 
    
    # Define all possible letters in 'ssl_history'
    all_characters = set('HCSVTXKRNYGFWUABDOPMIJLQhcsvtxkrnygfwuabdopmijlq')

    # Create binary columns for each character
    for char in all_characters:
        new_df[f'ssl_history_has_{char}'] = new_df['ssl_history'].apply(lambda x: 1 if char in x else 0)


    new_df = new_df.drop(columns='ssl_history')
    return new_df 

def create_broadcast_variable(new_df):
    # create broadcast variable
    # can have more than one broadcast address
    #255 is the broadcast address for ipv4 
    if 'id.resp_h' in new_df.columns:
        new_df['is_destination_broadcast'] = new_df['id.resp_h'].apply(lambda x: 1 if "255" in x[-3:] else 0) 
    return new_df

def create_direction_variable(new_df):
    #create traffic direction variable
    new_df['traffic_direction']        = new_df.apply(lambda x: get_traffic_direction(x['id.orig_h'], x['id.resp_h']), axis=1) 
    return new_df

def one_hot_encode(df, column_name):
    for col in column_name:
        if col in df.columns:
            df = pd.get_dummies(data=df, columns=[col])
    return df

def duration_to_numerical(new_df):
    # Convert duration to string
    new_df['duration'] = new_df['duration'].astype(str)
    # Extract the time portion (HH:MM:SS.mmmmmm) from the 'duration' column
    new_df['duration'] = new_df['duration'].str.extract(r'\d days (.*)')
    # Convert the time portion to a numerical format (float)
    new_df['duration'] = pd.to_timedelta(new_df['duration']).dt.total_seconds()
    return new_df 

def fill_na(new_df):
    
    #Fill Nans with 0s : duration, orig_bytes resp_bytes
    # Specify the columns you want to fill with zeros
    columns_to_fill_with_zeros = ['duration', 'orig_bytes', 'resp_bytes']
    # Check if columns exist; if not, create and fill with zeros
    for col in columns_to_fill_with_zeros:
        if col not in new_df.columns:
            new_df[col] = 0
    new_df[columns_to_fill_with_zeros] = new_df[columns_to_fill_with_zeros].fillna(0)
    
    #Fill Nans with 'Other' : service
    columns_to_fill_with_other = ['service']
    if 'service' in new_df.columns:
    # new_df['service'].fillna('other', inplace=True)
        new_df['service'] = new_df['service'].fillna('other')
        
    return new_df

def makedf_samecol(cols, new_df):
    #Create these columns if they are not present in the original df and fill them with 0s. 
    # Ensure that all the specified columns are present even if they are not present in the original df. 

    for col in cols:
        if col not in new_df.columns:
            new_df[col] = 0
    return new_df[cols]

def get_raw_conn(json_data_file):
    features = ["id.orig_h", "id.resp_h", "proto", "service", "duration", "conn_state", "local_orig","local_resp",
            "missed_bytes","history", "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes"]
    data_list = []
    for line in json_data_file.splitlines():
        # log_entry is now a single json log from the file
        log_entry = json.loads(line.strip())
        
        # Check if each feature is present in the log_entry
        feature_values = [log_entry.get(feature, None) for feature in features]
        data_list.append(feature_values)

    df = pd.DataFrame(data_list, columns=features)
    df = create_broadcast_variable(df)
    df = create_direction_variable(df)
    # Convert the boolean values in columns "local_orig" and "local_resp" to 1 and 0s
    df['local_orig'] = df['local_orig'].astype(int)
    df['local_resp'] = df['local_resp'].astype(int)
    
    return df 

def calculate_agg_feature_num(df, agg_feature, window_size):
  """
  This function adds a new column "{agg_feature}_{either mean, min, max, std, or var}" to the DataFrame.
  This column contains the aggregated features (mean/min/max/std/var/count/sum) of network flows within the past {window_size} seconds
  for each group with the same ['id.orig_h', 'id.orig_p', 'id.resp_h', 'id.resp_p'].

  Args:
      df: The pandas DataFrame containing network flow data.
      window_size: Size of the window for calculating the average (default: 5000 seconds).

  Returns:
      A new DataFrame with the added aggregated feautre columns.
  """
  # Convert timestamp to datetime
  # df['ts'] = datetime.fromtimestamp(df['ts']) #assumes timestamps are in the local machine's timezone. not suggested  
  df['ts'] = pd.to_datetime(df['ts'], unit='s') 
  df = df.set_index('ts') 
  # Calculate the aggregated feature for each group
  # to avoid NaN values, calculate the population standard deviation, specified with std(ddof=0)
  grp = ['id.orig_h', 'id.orig_p', 'id.resp_h', 'id.resp_p']
  df[f'{agg_feature}_mean_{window_size}'] = df.groupby(grp)[f'{agg_feature}'].transform(lambda x: x.rolling(f'{window_size}s', min_periods=1).mean())
  df[f'{agg_feature}_min_{window_size}'] = df.groupby(grp)[f'{agg_feature}'].transform(lambda x: x.rolling(f'{window_size}s', min_periods=1).min())
  df[f'{agg_feature}_max_{window_size}'] = df.groupby(grp)[f'{agg_feature}'].transform(lambda x: x.rolling(f'{window_size}s', min_periods=1).max())
  df[f'{agg_feature}_std_{window_size}'] = df.groupby(grp)[f'{agg_feature}'].transform(lambda x: x.rolling(f'{window_size}s', min_periods=1).std(ddof=0))
  df[f'{agg_feature}_var_{window_size}'] = df.groupby(grp)[f'{agg_feature}'].transform(lambda x: x.rolling(f'{window_size}s', min_periods=1).var(ddof=0))
  df[f'{agg_feature}_cnt_{window_size}'] = df.groupby(grp)[f'{agg_feature}'].transform(lambda x: x.rolling(f'{window_size}s', min_periods=1).count())
  df[f'{agg_feature}_sum_{window_size}'] = df.groupby(grp)[f'{agg_feature}'].transform(lambda x: x.rolling(f'{window_size}s', min_periods=1).sum())

  return df.reset_index()

#For feature such as  local_orig , port,... numerical but can be treated as categorical
def calculate_agg_feature_cat(df, agg_feature, window_size):
  """
  This function adds a new column "{agg_feature}_{either nunique or entropy}" to the DataFrame.
  This column contains the aggregated features (nunique/entropy) of network flows within the past {window_size} seconds
  for each group with the same ['id.orig_h', 'id.orig_p', 'id.resp_h', 'id.resp_p'].

  Args:
      df: The pandas DataFrame containing network flow data.
      window_size: Size of the window for calculating the average (default: 5000 seconds).

  Returns:
      A new DataFrame with the added aggregated feautre columns.
  """
  # Convert timestamp to datetime
  # df['ts'] = datetime.fromtimestamp(df['ts']) #assumes timestamps are in the local machine's timezone. not suggested  
  df['ts'] = pd.to_datetime(df['ts'], unit='s') 
  df = df.set_index('ts') 
  grp = ['id.orig_h', 'id.orig_p', 'id.resp_h', 'id.resp_p']
  df[f'{agg_feature}_nunique_{window_size}'] = df.groupby(grp)[f'{agg_feature}'].transform(lambda x: x.rolling(f'{window_size}s', min_periods=1).apply(lambda x: x.unique().shape[0]))
  df[f'{agg_feature}_entropy_{window_size}'] = df.groupby(grp)[f'{agg_feature}'].transform(lambda x: x.rolling(f'{window_size}s', min_periods=1).apply(lambda x: entropy(x.value_counts()))) 
  return df.reset_index()


#------------------Online Normalization------------------#
#TODO: def online_normalization(new_df):
# can be skipped for now, since kitnet has its own normalization.


