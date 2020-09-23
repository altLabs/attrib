import numpy as np
import pandas as pd
import ray
from ray.tune import Trainable
import datetime
import os
import sys
sys.path.append("../common/")
import torch
from common import mytorch


def torch_to_ray_trainable(
        ModelClass,
        loss_fn,
        OptClass,
        get_data_fn,
        train_step_period=300,
):

    class TrainableModule(Trainable):

        def _setup(self, config):
            self.config = config
            try:
                self.device = self.config['my_device']
            except:
                self.device = torch.device(
                    "cuda:0" if torch.cuda.is_available() else "cpu")
                self.config['my_device'] = self.device
            self.ModelClass = ModelClass
            self.OptClass = OptClass
            
            sys.stdout.flush()
            self.model = self.ModelClass(self.config)
            
            sys.stdout.flush()
            self.model.to(device=self.device)
            self.loss_fn = loss_fn
            self.optim = self.OptClass(
                self.model.parameters(), lr=self.config['lr'])
            self.get_data_fn = get_data_fn
            self.train_loader, self.val_loader = get_data_fn(**self.config)
            self.global_step = 0
            self.train_step_period = train_step_period
            self.ident = datetime.datetime.now().strftime("%d_%b_%Y_%I_%M_%S_%f%p")

        def _train(self):
            # Run your training op for n iterations
            
            components = self.model, self.loss_fn, self.optim, self.train_loader, self.val_loader

            train_loss, train_accuracy, val_loss, val_accuracy, self.global_step = mytorch.fit(
                self.train_step_period, *components, global_step=self.global_step, device=self.device)
            
            metrics = {
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "accuracy": val_accuracy,
                "validation_loss": val_loss,
                "neg_val_loss": -1 * val_loss,
            }
            return metrics

        def _stop(self):
            self.model = None
            self.train_loader = None
            self.val_loader = None

        def _save(self, checkpoint_dir):
            """
            """
            filepath = os.path.join(checkpoint_dir, self.ident)
            mytorch.torch_save(filepath, self.model, self.loss_fn,
                               self.optim, self.config, self.global_step)
            return filepath

        def _restore(self, checkpoint_path):
            """
            """
            print(f"IN restore, device is {self.device}")
            self.model, self.loss_fn, self.optim, self.train_loader, self.val_loader, self.config, self.global_step = mytorch.torch_load(
                checkpoint_path, self.ModelClass, self.OptClass, self.get_data_fn, device=self.device)
    return TrainableModule
