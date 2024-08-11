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
import threading

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def update_moving_average(ip_anomaly_dict, ip, anomaly_score, alpha=0.3, threshold=0.7):
    if ip not in ip_anomaly_dict:
        ip_anomaly_dict[ip] = {'count': 0, 'moving_average': 0, 'acl_active': False}
    ip_anomaly_dict[ip]['count'] += 1

    # Calculate the exponential moving average
    moving_average = alpha * anomaly_score + (1 - alpha) * ip_anomaly_dict[ip]['moving_average']
    ip_anomaly_dict[ip]['moving_average'] = moving_average

    # Check if the moving average crosses the threshold
    if moving_average > threshold:
        try:
            call_me(ip, 'add')
            ip_anomaly_dict[ip]['acl_active'] = True
            logger.info(f"ACL added for IP {ip}. Moving average: {moving_average}")
        except Exception as e:
            logger.error(f"Error adding ACL for IP {ip}: {e}")

    # Try to remove the ACL if the moving average falls below the threshold
    if moving_average <= threshold and ip_anomaly_dict[ip]['acl_active'] is True:
        try:
            call_me(ip, 'delete')
            ip_anomaly_dict[ip]['acl_active'] = False
            logger.info(f"ACL removed for IP {ip}. Moving average: {moving_average}")
        except Exception as e:
            logger.error(f"Error removing ACL for IP {ip}: {e}")

    return ip_anomaly_dict[ip]['moving_average']


def call_me(ip, action='add'):
    # Simulate adding or deleting the ACL
    logger.debug(f"Action {action} performed on IP {ip}.")


def get_all_acl():
    # TODO:here we get all ACLs as dicts from the switches
    acls = {}
    return acls


def maintain_states(ip_anomaly_dict):
    # Schedule the function to run again after 60 seconds
    threading.Timer(60, maintain_states, args=(ip_anomaly_dict,)).start()

    # Get the current state from the remote source
    remote_acls = get_all_acl()

    # Convert lists to sets for faster membership checking
    remote_acl_set = set(remote_acls.keys())
    active_local_acls = {ip for ip, data in ip_anomaly_dict.items() if data.get('acl_active', False)}

    # Determine ACLs that are in the remote set but not in the local set (to delete)
    to_delete = remote_acl_set - active_local_acls
    for acl in to_delete:
        call_me(acl, 'delete')

    # Determine ACLs that are in the local set but not in the remote set (to add)
    to_add = active_local_acls - remote_acl_set
    for acl in to_add:
        call_me(acl, 'add')


def main(log_path, threshold):
    global ip_anomaly_dict = {}
    # Schedule cleaning up every minute
    maintain_states(ip_anomaly_dict)

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
            logger.info(f"Resp IP: {id_resp_h}, Orig IP: {id_orig_h}, Anomaly Score: {anomaly_score}")

            # Update moving averages for both IPs
            resp_ip_avg = update_moving_average(ip_anomaly_dict, id_resp_h, anomaly_score, threshold=threshold)
            orig_ip_avg = update_moving_average(ip_anomaly_dict, id_orig_h, anomaly_score, threshold=threshold)
            logger.info(f"Updated Moving Average for Resp IP ({id_resp_h}): {resp_ip_avg}")
            logger.info(f"Updated Moving Average for Orig IP ({id_orig_h}): {orig_ip_avg}")

        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON: {json_part}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process log file and monitor anomaly scores.')
    parser.add_argument('--log_path', type=str, help='Path to the log file')
    parser.add_argument('--threshold', type=float, help='Threshold for the moving average')
    args = parser.parse_args()

    main(args.log_path, args.threshold)
