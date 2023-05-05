from torch import nn
import torch
import os
import sys
from skorch.callbacks import Callback
from copy import deepcopy

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

class MLP(nn.Module):
    def __init__(self, input_feature, layer_size, num_layers, activation, dropout_rate) -> None:
        torch.manual_seed(3407)
        
        super().__init__()

        blocks = [
            nn.Sequential(
                nn.Linear(layer_size if i > 0 else input_feature, layer_size),
                nn.BatchNorm1d(layer_size),
                activation(),
                nn.Dropout(dropout_rate)
            )
            for i in range(num_layers)
        ]

        self.layers = nn.Sequential(
            *blocks,
            nn.Linear(layer_size, 1),
        )

        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                m.bias.data.fill_(0.01)

        self.layers.apply(init_weights)

    def forward(self, x):
        return self.layers(x)
    
class CheckpointCallback(Callback):
    def on_epoch_end(self, net, dataset_train=None, dataset_valid=None, **kwargs):
        net.history.record('checkpoint', deepcopy(net.module_.state_dict()))