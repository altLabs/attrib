"""
nv_training.py

Train the CNN model based on Nielsen and Voigt (2018). 
This files uses the pytorch training environment (see README).

This was trained on a K80 GPU (p2.xlarge on AWS)

Run with ipython eg
ipython nv_training.py
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



device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")


datasets_dir = '../../../data/tts/'

expt_name = "nv_orig_params"
logdir = "./logs"
# Path to checkpoint from: None or a string
restore_path = None
logpath = os.path.join(logdir, expt_name)
get_ipython().system('mkdir {logpath}')
# Save a model checkpoint every these validation iterations
model_checkpoint_interval = 5

train_y = pickle.load( open( os.path.join(datasets_dir,"y_train_ord.pkl"), "rb" ) ).astype(np.int64)

# These parameters are copied from Nielsen and voigt
model_params = {'vocab_size': 5,
  'num_classes': 1314,
  'lr': 0.001,
  'batch_size': 8,
  'max_len': 8000,
  'filter_number': 128,
  'filter_len': 12,
  'num_dense_nodes': 64,
  'input_len': 16048,
  'TRAIN_X': os.path.abspath(os.path.join(dataset_dir,'train_x_no_nan.pkl')),
  'TRAIN_Y': os.path.abspath(os.path.join(dataset_dir,'y_train_ord.pkl')),
  'VAL_X': os.path.abspath(os.path.join(dataset_dir,'val_x_no_nan.pkl')),
  'VAL_Y': os.path.abspath(os.path.join(dataset_dir,'y_val_ord.pkl'))}

model_params['my_device'] = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")
model = mytorch.NVCNN(model_params)
model.to(device)


# In[ ]:


train_loader, val_loader = mytorch.get_NV_attrib_data(model_params['batch_size'], num_workers=4)
loss_func = F.cross_entropy
optim = torch.optim.Adam(model.parameters(),lr=model_params['lr'])
components = model, loss_func, optim, train_loader, val_loader


# In[ ]:


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

