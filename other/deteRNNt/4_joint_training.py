"""
4_joint_training.py

The final step in deteRNNt training: 
Now finetune the entire model, including the MLP and base sequence model, jointly.
We trained this for 300,300 steps to produce the final deteRNNt model.

This file uses the pytorch training environment. (view README) on an AWS p2.xlarge. 

Run with ipython e.g. 

ipython 4_joint_training.py
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
from sklearn.utils import class_weight
np.random.seed(44)
torch.manual_seed(44)


# ## This model was trained on subsequences. Predict probabilities for the classes as you slide along a window of the sequence, and average them. 


device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")

datasets_dir = '../../../data/'

bpe_models_dir = os.path.join(datasets_dir, 'bpe')
expt_name = "run_98_MLP_full_model"
logdir = "./logs"
# Path to checkpoint from: None or a string
restore_path = None
logpath = os.path.join(logdir, expt_name)
get_ipython().system('mkdir {logpath}')
# Save a model checkpoint every these validation iterations
model_checkpoint_interval = 5

model_stuff = torch.load(os.path.join(datasets_dir, "results/models/MLP_model1443300.pkl"))

model_params = model_stuff['params']
model_params['my_device'] = device
model_params['backprop'] = True
base = mytorch.myLSTMOutputHidden(model_params)
model = nn.Sequential(base,
                      nn.Linear(model_params['linear_layer_sizes'][-1] + 39, model_params['linear_layer_sizes'][-1]),
                      nn.ELU(),
                      nn.Dropout(),
                      nn.Linear(model_params['linear_layer_sizes'][-1], 1314),
)
model.load_state_dict(model_stuff['state_dict'])
model.to(device)


train_loader, val_loader = mytorch.get_attrib_bpe_data(model_params['batch_size'], modelpath=os.abspath(os.path.join(bpe_models_dir,'attrib_1000.model')), num_workers=4)
loss_func = F.cross_entropy
optim = torch.optim.Adam(model.parameters(),lr=.00005)
components = model, loss_func, optim, train_loader, val_loader


# I will do jank score logging in this case
global_step = 0
logfile = os.path.join(logpath, "logs.csv")
with open(logfile, "w+") as f:
    f.write(f"global_step,train_loss,train_accuracy,val_loss,val_accuracy\n")

try:
    i = 0
    while True:
        train_loss, train_accuracy, val_loss, val_accuracy, global_step = mytorch.fit(
            300,*components, global_step=global_step, device=device)
        # Log everything
        with open(logfile, "a") as f:
            line = f"{global_step},{train_loss},{train_accuracy},{val_loss},{val_accuracy}\n"
            f.write(line)
        if i % model_checkpoint_interval == 0:
            filepath = os.path.join(logpath, 'model' + str(global_step)+ '.pkl')
            mytorch.torch_save(filepath, model, loss_func,
                           optim, model_params, global_step)

        i += 1
except KeyboardInterrupt:
    print("Saw ctrl-C, saving before death")
    filepath = os.path.join(logpath, 'model' + str(global_step)+ '.pkl')
    mytorch.save(filepath, model, loss_func,
                           optim, model_params, global_step)

