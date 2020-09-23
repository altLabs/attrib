"""
3_metadata_MLP_topmodel.py

The third step in deteRNNt training: 
Concatenate metadata with the final hidden layer of step 2, and train a 
2-layer multi-layer perceptron (MLP).

This file uses the pytorch training environment. (view README) on an AWS p2.xlarge. 

Run with ipython e.g. 

ipython 3_metadata_topmodel.py
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

device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")

datasets_dir = '../../../data/'

bpe_models_dir = os.path.join(datasets_dir, 'bpe')
expt_name = "run_98_MLP_topmodel"
logdir = "./logs"
# Path to checkpoint from: None or a string
restore_path = None
logpath = os.path.join(logdir, expt_name)
get_ipython().system('mkdir {logpath}')
# Save a model checkpoint every these validation iterations
model_checkpoint_interval = 5


model_stuff = torch.load(os.path.join(datasets_dir, "results/models/model201300.pkl"))


model_params = model_stuff['params']
model_params['my_device'] = device
model_params['backprop'] = False
base = mytorch.myLSTMOutputHidden(model_params)
base.load_state_dict(model_stuff['state_dict'], strict=False)
model = nn.Sequential(base,
                      nn.Linear(model_params['linear_layer_sizes'][-1] + 39, model_params['linear_layer_sizes'][-1]),
                      nn.ELU(),
                      nn.Dropout(),
                      nn.Linear(model_params['linear_layer_sizes'][-1], 1314),
)
model.to(device)


train_loader, val_loader = mytorch.get_attrib_bpe_data(model_params['batch_size'], modelpath=os.abspath(os.path.join(bpe_models_dir,'attrib_1000.model')), num_workers=4)
loss_func = model_stuff['loss_func']
optim = torch.optim.Adam(model.parameters(),lr=.0001)
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

