"""
2_long_training.py

The second step in deteRNNt training: 
Run the best performing model from hyperband search to plateau.

This file uses the pytorch training environment. (view README) on an AWS p2.xlarge. 

Run with ipython e.g. 

ipython 2_long_training.py
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data
import numpy as np
import pandas as pd
import sentencepiece as sp
import pickle
import ray
import sys
import random
import os
sys.path.append("../../common/")
from common import mytorch
from common import data_io_utils
np.random.seed(44)
torch.manual_seed(44)


datasets_dir = '../../../data'


bpe_models_dir = os.path.join(datasets_dir, 'bpe')


expt_name = "run_98_from_first_bpe"
logdir = "./logs"
# Path to checkpoint from: None or a string
restore_path = None
logpath = os.path.join(logdir, expt_name)
get_ipython().system('mkdir {logpath}')
# Save a model checkpoint every these validation iterations
model_checkpoint_interval = 5



if restore_path is None:
    get_ipython().system('rm {logdir}/{expt_name}/* -r')



params = {
    'dropout_layers': [None],
    'activation': "relu",
    'vocab_size': 5,
    'num_lstm_hidden': 6,
    'num_lstm_layers': 3,
    'bidir': True,
    'other_features_size': 39,
    'linear_layer_sizes': [100],
    'num_classes': 1314,
    'lr': .0005,
    'batch_size': 8,
    'max_len': 128,
    'include_metadata': False,
    'modelpath': None,
    'embed_dim': 2
}
params = {**params, **mytorch.get_attrib_paths()}


# Copy the dictionary of run-specific parameters here
variable_config = {
    "num_workers":6, # Should speed up data fetching on a multi-CPU machine
    "activation": "elu",
    "batch_size": 32,
    "bidir": True,
    "dropout_layers": [
        0.5
    ],
    "embed_dim": 200,
    "include_metadata": False,
    "linear_layer_sizes": [
        1000
    ],
    "lr": 0.0001,
    "max_len": 512,
    "modelpath": os.abspath(os.path.join(bpe_models_dir, "attrib_1000.model")),
    "num_classes": 1314,
    "num_lstm_hidden": 128,
    "num_lstm_layers": 2,
    "other_features_size": 39,
    "other_linear_depth": 2,
    "vocab_size": 1000
}


full_config = {**params, **variable_config}

# This uses the exposed ray trainable interface without the ray garbage
trainable = mytorch.RaylessTrainable(
    full_config,
    mytorch.myLSTM,
    F.cross_entropy,
    torch.optim.Adam,
    mytorch.get_attrib_bpe_data
)


if restore_path is not None:
    trainable._restore(restore_path)

# I will do jank score logging in this case
logfile = os.path.join(logpath, "logs.csv")
with open(logfile, "w+") as f:
    f.write(f"global_step,train_loss,train_accuracy,val_loss,val_accuracy\n")

try:
    i = 0
    while True:
        metrics = trainable._train()
        # Log everything
        with open(logfile, "a") as f:
            line = f"{metrics['global_step']},{metrics['train_loss']},{metrics['train_accuracy']},{metrics['val_loss']},{metrics['val_accuracy']}\n"
            f.write(line)
        if i % model_checkpoint_interval == 0:
            trainable._save(logpath)

        i += 1
except KeyboardInterrupt:
    print("Saw ctrl-C, saving before death")
    trainable._save(logpath)
