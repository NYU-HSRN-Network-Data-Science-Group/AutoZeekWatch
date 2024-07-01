# AutoZeekWatch

AutoZeekWatch is a real-time, modular, multiprocessed, configurable A.I. anomaly detector for [Zeek](https://zeek.org/) logs. AutoZeekWatch enables you to generate anomaly scores for Zeek logs im real time, and correlate them with the initial 5-tuple and Zeek UID for downstream analysis, automated mitigation, and more. 

## Table of Contents

* Features
* Zeek Data Format
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

## Zeek Data Format

AutoZeekWatch requires Zeek data to be in JSON format. In your `local.zeek` add the following:

```
@load policy/tuning/json-logs.zeek
```

If you have logs in standard TSV format, you can convert them to JSON using [Zeek Analysis Tools](https://github.com/SuperCowPowers/zat). 

## Installation and Training

On your NIDS node, run:

```
git clone https://github.com/NYU-HSRN-Network-Data-Science-Group/AutoZeekWatch.git
cd AutoZeekWatch/
pip install -r requirements.txt
python AutoZeekWatch/train.py --log-dir <ZEEK_LOG_DIR> --modules <MODULE1> <MODULE2> 
```

If you would like to train the model in the background:

```
python AutoZeekWatch/train.py --log-dir <ZEEK_LOG_DIR> --modules <MODULE1> <MODULE2> &
```

You can list your current Zeek log directory with:

```
zeekctl config | grep logdir
```

## Examples

### Train a Model on Connection Data and SSH Data

```
python train.py --log-dir <PATH/TO/LOGS> --modules CONN SSH &
```

### Start Inference on Incoming Connection Data

```
python infer.py --log-dir <PATH/TO/LOGS> --modules CONN
```

