"""
1_hyperband_search.py

The first step in deteRNNt training: 
search model and hyperparameter configurations for the best configuration.

Note that the exact hyperparameters may not be searched by reproducing this
run- we have been unable to determine whether ray uses the numpy and torch seeds
instead of something else.

This file uses the pytorch training environment. (view README) on an AWS p2 series
machine (more GPUs return approximately linear speedups).

Run with ipython e.g.:
ipython 1_hyperband_search.py
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
from common.torch_interface import torch_to_ray_trainable
from common import data_io_utils
np.random.seed(44)
torch.manual_seed(44)


datasets_dir = '../../../data/'

bpe_models_dir = os.path.join(datasets_dir, '/bpe')



expt_name = "hyperband_lstm"
ray_results = "./ray_results"
restore = False
smoke = False
# ray_results


# In[3]:


if not restore:
    get_ipython().system('rm logs/* -r')
    get_ipython().system('rm {ray_results}/{expt_name}/* -r')
    get_ipython().system('rm tmp/* -r')


# In[4]:


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


# In[5]:


Trainable = torch_to_ray_trainable(
    mytorch.myLSTM,
    F.cross_entropy,
    torch.optim.Adam,
    mytorch.get_attrib_bpe_data
)


# In[6]:


variable_config = {
    "lr": ray.tune.sample_from(lambda spec: random.choice([.00001, .00005, .00075, .0001, .00025, .0005, .001, .0025, .005, .01])),
    "batch_size": ray.tune.sample_from(lambda spec: random.choice([4, 8, 16, 32, 64, 128, 256])),
    "bidir": ray.tune.sample_from(lambda spec: random.choice([True, False])),
    "num_lstm_hidden": ray.tune.sample_from(lambda spec: random.choice([8, 12, 16, 32, 48, 64, 128, 256, 512])),
    "num_lstm_layers": ray.tune.sample_from(lambda spec: random.choice([1, 2, 4, 8])),
    "other_linear_depth": ray.tune.sample_from(lambda spec: random.choice([0, 0, 0, 1, 1, 2, 3])),

    "linear_layer_sizes": ray.tune.sample_from(
        lambda spec: [random.choice([50, 75, 100, 200, 300, 500, 1000])] + [
            random.choice([50, 75, 100, 200, 300, 500, 1000]) for _ in range(spec.config.other_linear_depth)
        ]
    ),
    "dropout_layers": ray.tune.sample_from(
        lambda spec: [random.choice([None, None, .25, .5])] + [
            random.choice([None, None, .25, .5]) for _ in range(spec.config.other_linear_depth)
        ]
    ),
    "activation": ray.tune.sample_from(lambda spec: random.choice(["relu", "relu", "relu", "selu", "elu"])),
    "max_len": ray.tune.sample_from(lambda spec: random.choice([8, 64, 128, 256, 512, 512, 512, 512, 512, 512, 1028, 2046])),
    'vocab_size': ray.tune.sample_from(lambda spec: random.choice([5, 100, 1000, 5000, 10000])),
    'modelpath': ray.tune.sample_from(
        lambda spec: None if spec.config.vocab_size == 5 else os.path.join(
            bpe_models_dir,
            random.choice(
                [f'attrib_{spec.config.vocab_size}.model',
                 f'attrib_sp_{spec.config.vocab_size}.model']
            ))
    ),
    'embed_dim': ray.tune.sample_from(
        lambda spec: 2 if spec.config.vocab_size == 5 else random.choice([5, 10, 30, 50, 100, 200])),


}
smoke_config = {
    "lr": ray.tune.sample_from(lambda spec: random.choice([.0001, .01, .005, .0025]))
}


# In[7]:


ray.tune.register_trainable("pytorch_hyperband", Trainable)


# In[8]:


ray.shutdown()

if smoke:
    full_config = {**params, **smoke_config}
    exp = ray.tune.Experiment(
        name="smoke_test",
        run="pytorch_hyperband",
        num_samples=3,
        stop={"training_iteration": 9999},
        config=full_config,
        resources_per_trial={"cpu": 3}
    )
    ray.init(redis_max_memory=1 * 10**9,
             object_store_memory=1 * 10**9,
             temp_dir="./tmp",
             )
    hyperband = ray.tune.schedulers.AsyncHyperBandScheduler(
        time_attr="training_iteration",
        reward_attr="accuracy",
        max_t=5,
        grace_period=2)

else:
    full_config = {**params, **variable_config}
    exp = ray.tune.Experiment(
        name=expt_name,
        run="pytorch_hyperband",
        num_samples=300,
        stop={"training_iteration": 9999},
        config=full_config,
        resources_per_trial={"cpu": 1, "gpu": .24},
        local_dir=ray_results
    )
    ray.init(redis_max_memory=5 * 10**9,
             object_store_memory=10 * 10**9,
             temp_dir="./tmp",
             )
    hyperband = ray.tune.schedulers.AsyncHyperBandScheduler(
        time_attr="training_iteration",
        reward_attr="accuracy",
        max_t=300,
        grace_period=10)


# In[9]:


ray.tune.run_experiments(exp, scheduler=hyperband, resume=restore)
