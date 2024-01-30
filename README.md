# Network Intrusion Detection System (NIDS)

## Purpose
This repository is dedicated to developing a Network Intrusion Detection System (NIDS) utilizing unsupervised machine learning techniques such as KitNET, Autoencoder, and Isolation Forest.

## Data Description
The input data for this system is Zeek conn logs. The data is unstructured, with variations in columns across different instances.

## Code Description

1. **data_preprocess.py:** This script preprocesses data specifically for the KitNET model.
2. **train_kitnet.py:** Contains code for training a KitNET model on HSRN data, using parameters from the best model. Use the argument to specify the date of the data to be trained on, for example, 2023-11-19.
3. **pred_kitnet.py:** This script processes data from the "current" folder, which stores new data, and preprocesses it for the KitNET model.