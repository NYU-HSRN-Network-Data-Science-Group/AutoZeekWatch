"""
This file reads the anomalies log produced by infer.py and dynamically 
changes the ACL for ERSPAN via OpenConfig REST API.

Authors: 
- Zoe Hsu <wh2405@nyu.edu> 
- Olive Song <js10417@nyu.edu> 
- Diego Lopez <dtl310@nyu.edu>
- Zihang Xia <zx961@nyu.edu>
"""

import tailer
import requests
from requests.auth import HTTPBasicAuth
import logging
import json
import re
import argparse

def update_moving_average(ip_anomaly_dict, ip, anomaly_score):
  if ip not in ip_anomaly_dict:
      ip_anomaly_dict[ip] = {'total_score': 0, 'count': 0}

  ip_anomaly_dict[ip]['total_score'] += anomaly_score
  ip_anomaly_dict[ip]['count'] += 1
  moving_average = ip_anomaly_dict[ip]['total_score'] / ip_anomaly_dict[ip]['count']

  return moving_average

def call_me(x,y):
    print("Crossed threshold")


def main(log_path, threshold):
    ip_anomaly_dict = {}
    for line in tailer.tail(open(log_path)):
        match = re.search(r'{.*}', line)
        if not match:
            continue 
        json_part = match.group(0).strip()
        json_part = re.sub(r"'", '"', json_part)
        try:
            log_data = json.loads(json_part)
            id_resp_h = log_data.get('id.resp_h')
            id_orig_h = log_data.get('id.orig_h')
            anomaly_score = log_data.get('anomaly_score')
            print(f"Resp IP: {id_resp_h}, Orig IP: {id_orig_h}, Anomaly Score: {anomaly_score}")
            # Update moving averages for both IPs
            resp_ip_avg = update_moving_average(ip_anomaly_dict, id_resp_h, anomaly_score)
            orig_ip_avg = update_moving_average(ip_anomaly_dict, id_orig_h, anomaly_score)
            print(f"Updated Moving Average for Resp IP ({id_resp_h}): {resp_ip_avg}")
            print(f"Updated Moving Average for Orig IP ({id_orig_h}): {orig_ip_avg}")
            # Check if the moving average crosses the threshold
            if resp_ip_avg > threshold:
                # call_me is just a placeholder for now
                call_me(id_resp_h, resp_ip_avg)
            if orig_ip_avg > threshold:
                call_me(id_orig_h, orig_ip_avg)
        except json.JSONDecodeError:
            print("Failed to parse JSON:", json_part)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process log file and monitor anomaly scores.')
    parser.add_argument('--log_path', type=str, help='Path to the log file')
    parser.add_argument('--threshold', type=float, help='Threshold for the moving average')
    args = parser.parse_args()

    main(args.log_path, args.threshold)