"""
deteRNNt_inference_logits.py

Produces logits of predictions by subsampling 100 subsequences from each 
plasmid, running them through either the classic deteRNNt model or one trained
with no metadata. 

Need to specify which model by uncommenting model loading blocks (defaults to
the classic deteRNNt model. The data to perform inference on is also specified
below.

Finally, the logits are saved in an output .npy file, defined at bottom.

* This is intended for use on an NVIDIA GPU. In our case, p2.xlarge K80 on AWS.
This file uses the pytorch training environment. (view README)


Run with ipython e.g. 

ipython deteRNNt_inference_logits.py
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

datasets_dir = '../../../data'

bpe_models_dir = os.path.join(datasets_dir, 'bpe')

# for sequence only
"""
model_stuff = torch.load(os.path.join(datasets_dir, "results/models/model201300.pkl"))
"""

# for metadata

model_stuff = torch.load(os.path.join(datasets_dir, "results/models/MLP_full_with_metadata_model300300.pkl"))



# Input the test or validation data
data = pickle.load( open( os.path.join(datasets_dir,'tts/val_x_no_nan.pkl'), "rb" ) )
data.head()

num_rows = len(data)
# In[4]:

device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")
model_stuff


# In[5]:
# For no metadata, uncomment this block, ow use the below 
"""
model_params = model_stuff['params']
print(model_params)
# model_params['my_device'] = torch.device('cpu')
model = mytorch.myLSTM(model_stuff['params'])
model.load_state_dict(model_stuff['state_dict'])
model.to(device)
"""

# Uncomment this block for use with the full deteRNNt model
model_params = model_stuff['params']
model_params['my_device'] = device
model_params['backprop'] = False
base = mytorch.myLSTMOutputHidden(model_params)
model = nn.Sequential(base,
                      nn.Linear(model_params['linear_layer_sizes'][-1] + 39, model_params['linear_layer_sizes'][-1]),
                      nn.ELU(),
                      nn.Dropout(),
                      nn.Linear(model_params['linear_layer_sizes'][-1], 1314),
)
model.load_state_dict(model_stuff['state_dict'])
model.to(device)

# In[6]:


sp_model = model_stuff['params']['modelpath']
if sp_model:
    # Hardwiring for a test locally
    sp_model = os.path.join(bpe_models_dir,"attrib_1000.model")
    processor = sp.SentencePieceProcessor()
    processor.Load(sp_model)


# In[7]:


max_len = model_stuff['params']['max_len']


# In[8]:


def get_subsequences(seq, length=max_len, max_num_subseqs=None):
    """
    Returns all subsequences of seq at least as long as length.
    If no such sequences exist, returns the original. 
    If max_num_subseqs is not None, then number of subseqs returned
    are randomly subsampled to that number
    """
    if len(seq) <= length:
        return [seq]
    else:
        subs = [seq[i: i + length] for i in range(len(seq) - length)]
        if (max_num_subseqs is not None) and len(subs) > max_num_subseqs:
            return random.sample(subs, max_num_subseqs)
        else:
            return subs
            


# In[9]:


predictions = []
test = data.iloc[:10,:]
with torch.no_grad():
    for i,seq in enumerate(data['sequence']):
        seq = processor.EncodeAsIds(seq)
        print(f"This seq is length {len(seq)}")
        sub_predictions = []
        num_subseqs = 0
        for subseq in get_subsequences(seq, max_num_subseqs=100):
            logits = model(([len(subseq)], torch.LongTensor([subseq]), torch.Tensor([data.iloc[i,1:]])))
            #print("Predicted logits")
            #probs = F.softmax(logits,dim=1).data.cpu().numpy()
            sub_predictions.append(logits.cpu().numpy())
            num_subseqs += 1
        print(f'there were {num_subseqs} in that seq')
        predictions.append(np.mean(np.array(sub_predictions), axis=0))
        


# In[10]:


np.array(predictions).shape


# In[11]:


np.sum(np.array(predictions), axis=2)


# In[14]:


working = np.reshape(predictions, (num_rows,1314))


# In[16]:


working.shape


# In[17]:


np.sum(np.array(working), axis=1)


# In[18]:


# This looks good. Lets save it.
np.save(os.path.join(datasets_dir, "results/VAL_LOGITS_100_sub_nometadata_201300.npy"), working)

