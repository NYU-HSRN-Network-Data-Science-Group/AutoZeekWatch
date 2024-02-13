"""
This script is used to preprocess the data for the kitnet model.
Example command:
root@worker19:/opt/zeek/kitnet# python3 data_preprocess.py 2023-11-18

Check the column names of the output file by using the following command:
head -n 1 df_2023-11-18.csv

There should be 46 columns in the output file. 
Different from the iot23 (no_aggre) dataset, it contains service_ntp

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
# TODO: is there a better way to handle multi-file logging aside from spamming these everywhere?
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s)')

def preprocess_json(json_batch):
    """
    This function receives a json batch from the main control flow of the train 
    functions. It should convert the json_batch to a numpy 2D array, apply necessary transformations,
    then return it. 

    Note: the input is only one unzipped json file. 
    """
    # TODO: add the featureset here 
    # TODO: should we move this feature set somewhere else?
    features = ['id.orig_p', "id.resp_p", "proto", "conn_state", "missed_bytes",
                "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes"]
    # add the following features ['duration', 'history']
    # TODO: @olive please run the script as is, it should work.
    # However, some log records in json do not have duration or history fields.
    # Please catch this error, and if there is no duration, add a duration of 0 to the record. 
    # If there is no history, add a history, with the value "N"
    data_list = []
    for line in json_batch.splitlines():
        # log_entry is now a single json log from the file
        log_entry = json.loads(line.strip())
        data_list.append([log_entry[feature] for feature in features])
    # np_arr = np.array(data_list)
    
    # TODO: apply transformations based on last semesters work
    #Re-use the preprocess function from last sem by Zoe. 
    #TODO: optimize the code via removing pandas
    new_df = pd.DataFrame(data_list, columns=features) 
    #Fill NaNs with 0s : duration, orig_bytes resp_bytes, if there are no columns, create one and fill with 0s 
    new_df = fill_na(new_df) 
    # Drop unnecessary columns 
    new_df = drop_columns(new_df, ['ts','uid','local_orig', 'local_resp'])
    
    # create history, broadcast, traffic_direction variables
    new_df = create_history_variable(new_df)
    new_df = create_broadcast_variable(new_df)
    new_df = create_direction_variable(new_df)

    # one hot encode categorical variables
    column_name = ['conn_state', "proto", "traffic_direction" , "service"]
    for col in column_name:
        if col in new_df.columns:
            new_df = one_hot_encode(new_df, [col])
    # new_df = new_df.drop(columns=['id.orig_h', 'id.resp_h'])

    new_df = drop_columns(new_df, ['id.orig_h', 'id.resp_h'])

    # make sure the columns are the same as the original df
    new_df = makedf_samecol(new_df)
    # new_df = new_df.drop(columns=['orig_l2_addr','resp_l2_addr'])
    new_df = drop_columns(new_df, ['orig_l2_addr','resp_l2_addr'])

    # Convert DataFrame to NumPy array
    np_arr = new_df.to_numpy()# np_arr is now a numpy 2D array
    
    print(np_arr[:5]) 
    print(np_arr.shape)  
    logging.info("Hello from preprocess_json. Please implement me :)")
    return np_arr


#------------------Dataframe Prep------------------#
def merge_logs(date):
    # This function is used to take in a date and merge all the logs for that date into a single dataframe.
    df_merged = pd.DataFrame()

    # Iterate over the hours and unzip files
    for hour in range(24):
        # Create the file paths
        log_file_path = f"../logs/{date}/conn.{hour:02d}:00:00-{(hour + 1) % 24:02d}:00:00.log.gz"
        unzipped_file_path = f"../logs/{date}/conn.{hour:02d}:00:00-{(hour + 1) % 24:02d}:00:00.log"

        # Solution-1: Unzip the file
        unzip_command = f"gunzip -k {log_file_path}"
        subprocess.run(unzip_command, shell=True)

        # #Solution-2: 
        # # Unzip the file
        # with gzip.open(log_file_path, 'rt') as f_in:
        #     with open(unzipped_file_path, 'w') as f_out:
        #         f_out.write(f_in.read())

        # Read the unzipped log file into a DataFrame
        df_hour = pd.read_json(unzipped_file_path, lines=True)

        # Append the data to the merged DataFrame
        df_merged = pd.concat([df_merged, df_hour], axis = 0, ignore_index=True)

        # Remove the unzipped log file
        os.remove(unzipped_file_path)

    return df_merged

#TODO : ask Diego. 
# df has these columns: ts,uid,id.orig_h,id.orig_p,id.resp_h,id.resp_p,proto,duration,orig_bytes,resp_bytes,conn_state,local_orig,local_resp,missed_bytes,history,orig_pkts,orig_ip_bytes,resp_pkts,resp_ip_bytes,orig_l2_addr,resp_l2_addr,service  
# Which has orig_l2_addr,resp_l2_addr that are not in the zeek iot23 dataset. 
# The zeek iot23 dataset has these columns:ts	uid	id.orig_h	id.orig_p	id.resp_h	id.resp_p	proto	service	 duration	orig_bytes	resp_bytes	conn_state	local_orig	local_resp	missed_bytes	history	orig_pkts	orig_ip_bytes	resp_pkts	resp_ip_bytes	tunnel_parents	label	detailed_label
# The zeek iot23 dataset tunnel_parents,label,detailed_label are not in the df.


#TODO : create a function that takes in multiple dates and returns a dataframe


#------------------Preprocessing------------------#
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
    octets = [int(x) for x in ip_str.split(".")]
    if octets[0] == 10 \
            or (octets[0] == 172 and 16 <= octets[1] <= 31) \
            or (octets[0] == 192 and octets[1] == 168):
        return True
    else:
        return False

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

    # if 'history'in new_df.columns:
    #     new_df = new_df[new_df['history'].notna()].copy() #filters out rows where the 'history' column is null
    #     new_df['history_has_S'] = new_df['history'].apply(lambda x: 1 if "S" in x else 0)
    #     new_df['history_has_h'] = new_df['history'].apply(lambda x: 1 if "h" in x else 0)
    #     new_df['history_has_A'] = new_df['history'].apply(lambda x: 1 if "A" in x else 0)
    #     new_df['history_has_D'] = new_df['history'].apply(lambda x: 1 if "D" in x else 0)
    #     new_df['history_has_a'] = new_df['history'].apply(lambda x: 1 if "a" in x else 0)
    #     new_df['history_has_d'] = new_df['history'].apply(lambda x: 1 if "d" in x else 0)
    #     new_df['history_has_F'] = new_df['history'].apply(lambda x: 1 if "F" in x else 0)
    #     new_df['history_has_f'] = new_df['history'].apply(lambda x: 1 if "f" in x else 0)
    #     new_df = new_df.drop(columns='history')
    # else: 
    #     new_df['history_has_S'] = 0
    #     new_df['history_has_h'] = 0
    #     new_df['history_has_A'] = 0
    #     new_df['history_has_D'] = 0
    #     new_df['history_has_a'] = 0
    #     new_df['history_has_d'] = 0
    #     new_df['history_has_F'] = 0
    #     new_df['history_has_f'] = 0
    
    if 'id.orig_h'in new_df.columns:
        new_df = new_df[new_df['id.orig_h'].str.contains("::") == False]
    return new_df 

def create_broadcast_variable(new_df):
    # create broadcast variable
    #255 is the broadcast address for ipv4(#TODO : ask Diego)
    if 'id.resp_h' in new_df.columns:
        new_df['is_destination_broadcast'] = new_df['id.resp_h'].apply(lambda x: 1 if "255" in x[-3:] else 0) 
    return new_df

def create_direction_variable(new_df):
    #create traffic direction variable
    if 'traffic_direction' in new_df.columns:
        new_df['traffic_direction']        = new_df.apply(lambda x: get_traffic_direction(x['id.orig_h'], x['id.resp_h']), axis=1) 
    return new_df

def one_hot_encode(df, column_name):
    new_df = pd.get_dummies(data=df, columns=column_name)
    return new_df

def duration_to_numerical(new_df):
    # Convert duration to string
    new_df['duration'] = new_df['duration'].astype(str)
    # Extract the time portion (HH:MM:SS.mmmmmm) from the 'duration' column
    new_df['duration'] = new_df['duration'].str.extract(r'\d days (.*)')
    # Convert the time portion to a numerical format (float)
    new_df['duration'] = pd.to_timedelta(new_df['duration']).dt.total_seconds()
    return new_df 


#TODO: create a function that takes in a dataframe and perform the preprocessing steps on it
def preprocess(new_df):
    
    # Drop unnecessary columns 
    columns_to_drop = ['ts','uid','local_orig', 'local_resp']
    new_df.drop(columns_to_drop, axis=1, inplace=True)

    # create history, broadcast, traffic_direction variables
    new_df = create_history_variable(new_df)
    new_df = create_broadcast_variable(new_df)
    new_df = create_direction_variable(new_df)

    # one hot encode categorical variables
    #TODO : discuss with Diego, if there's a better way to do this. since, input dataset may have different conn state, that means the columns would be different. 
    column_name = ['conn_state', "proto", "traffic_direction" , "service"]
    for col in column_name:
        if col in new_df.columns:
            new_df = one_hot_encode(new_df, [col])
    new_df = new_df.drop(columns=['id.orig_h', 'id.resp_h'])

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

def makedf_samecol(new_df):
    #Create these columns if they are not present in the original df and fill them with 0s. 
    # Ensure that all the specified columns are present even if they are not present in the original df. 
    cols = ['conn_state_OTH', 'conn_state_REJ','conn_state_RSTO', 'conn_state_RSTOS0', 'conn_state_RSTR','conn_state_RSTRH', 
        'conn_state_S0', 'conn_state_S1', 'conn_state_S2','conn_state_S3', 'conn_state_SF', 'conn_state_SH', 'conn_state_SHR',
        'proto_tcp', 'proto_udp',
        'service_dhcp', 'service_dns','service_http', 'service_irc','service_ntp',
        'service_other', 'service_ssh','service_ssl',
        'traffic_direction_external','traffic_direction_incoming', 
        'traffic_direction_internal','traffic_direction_outgoing']
    for col in cols:
        if col not in new_df.columns:
            new_df[col] = 0
    return new_df

#------------------Online Normalization------------------#
#TODO: def online_normalization(new_df):
# can be skipped for now, since kitnet has its own normalization.



def main(date_argument):
    df_merged = merge_logs(date_argument)
    
    #df_merged = duration_to_numerical(df_merged)
    df_merged_filled = fill_na(df_merged)
    
    # run preprocess 
    df_merged_preprocessed = preprocess(df_merged_filled)

    # make sure the columns are the same as the original df
    df_final = makedf_samecol(df_merged_preprocessed)
    df_final = df_final.drop(columns=['orig_l2_addr','resp_l2_addr'])
    print("The dataframe is processed successfully, with the shape of ", df_final.shape)
    
    # run online normalization 
    # can be skipped for now, since kitnet has its own normalization.

    # Save the merged DataFrame to a CSV file
    df_final.to_csv(f"df_{date_argument}.csv", index=False)

    return df_final


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python data_preprcess.py <date>")
        sys.exit(1)

    date_argument = sys.argv[1]
    df_final = main(date_argument)


