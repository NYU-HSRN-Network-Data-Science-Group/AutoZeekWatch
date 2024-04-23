# AutoZeekWatch

AutoZeekWatch is a real-time, modular, configurable A.I. anomaly detector for [Zeek](https://zeek.org/) logs. AutoZeekWatch enables you to generate anomaly scores for Zeek logs im real time, and correlate them with the initial 5-tuple and Zeek UID for downstream analysis, automated mitigation, and more. 

## Table of Contents

* Features
* Installation
* Examples

## Features

AutoZeekWatch functions in two distinct phases, **training** and **inference**. Under the hood, [KitNET](https://pysad.readthedocs.io/en/latest/generated/pysad.models.KitNet.html), a ensemble of autoencoders, is used to generate anomaly scores for individual logs in a unsupervised manner. 

During **training**, the model must learn the *normal* distribution from provided data. The user is expected to provide a directory where historical, *normal* (not malicious) logs are stored. The model then learns this distribution. 

During **inference**, the model provides a score of how anomalous a given log is to the distribution learned from training. This score along with the 5-tuple (Source IP, Destination IP, Source Port, Destination Port, Proto) is then dumped to a file which can be used for downstream tasks or alerting. 

It is possible to specify different zeek log types to train on and perform inference on. Currently, the following are available:

- Connection
- HTTP
- DNS
- SSH
- SSL

These can be used modularly, one, many, or all can be used at once. 

## Installation

...

## Examples

### Train a Model on Connection Data

```
python train.py --log-dir <PATH/TO/LOGS> --modules CONN
```

### Start Inference on Incoming Connection Data

```
python infer.py --log-dir <PATH/TO/LOGS> --modules CONN
```