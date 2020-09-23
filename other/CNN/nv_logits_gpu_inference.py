"""
nv_logits_gpu_inference.py

Run the CNN model based on Nielsen and Voigt (2018). Uses a dummy y path to use
the same data loader.

This files uses the pytorch training environment (see README).
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


# ## This model was trained on subsequences. Predict probabilities for the classes as you slide along a window of the sequence, and average them. 

datasets_dir = '../../../data/'



model_stuff = torch.load(os.path.join(datasets_dir, "CNN/nv_exact_params_our_data_model787800.pkl"))

x_path = os.path.join(datasets_dir,'tts/test_x_no_nan.pkl')
y_path =  os.path.join(datasets_dir,'tts/test_y_dummy.pkl')

# Where to save file
predict_name = os.path.join(datasets_dir,"results/TEST_LOGITS_nv_787800.npy"



data = pickle.load( open(x_path , "rb" ) )
data.head()
true = pickle.load( open(y_path, "rb" ) )
true

dataset = mytorch.NVAttribDataset(x_path,y_path)
num_rows = len(data)


device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")
model_stuff




model_params = model_stuff['params']
print(model_params)
# model_params['my_device'] = torch.device('cpu')
model = mytorch.NVCNN(model_stuff['params'])
model.load_state_dict(model_stuff['state_dict'])
model.to(device)
model.eval()



max_len = model_stuff['params']['max_len']


predictions = []
test = data.iloc[:10,:]
with torch.no_grad():
    for i,(seq, _) in enumerate(dataset): # Using Dataset.__get__item() here
        if i % 100 == 0:
            print(f"processed {i} examples")
        logits = model(torch.Tensor([seq]))
        predictions.append(logits.cpu().numpy())


print(f'Predictions shape {np.array(predictions).shape}')




np.sum(np.array(predictions), axis=2)




working = np.reshape(predictions, (num_rows,1314))




working.shape




np.sum(np.array(working), axis=1)




# This looks good. Lets save it.
np.save(os.path.join(datasets_dir, predict_name), working)
print("Saved!")

