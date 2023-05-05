from auton_survival.models.cph import DeepCoxPH
import numpy as np
import torch
from torch import nn
from auton_survival.models.cph.dcph_utilities import train_dcph

class DropoutDeepCoxPHTorch(nn.Module):
    def _init_coxph_layers(self, lastdim):
        self.expert = nn.Linear(lastdim, 1, bias=False)

    def __init__(self, inputdim, layers=None, optimizer='Adam', dropout_rates=None, activation="ReLU6"):

        super(DropoutDeepCoxPHTorch, self).__init__()

        self.optimizer = optimizer

        if layers is None: layers = []
        self.layers = layers

        if dropout_rates is None: dropout_rates = []

        if activation == 'ReLU6':
            act = nn.ReLU6()
        elif activation == 'ReLU':
            act = nn.ReLU()
        elif activation == 'SeLU':
            act = nn.SELU()
        elif activation == 'Tanh':
            act = nn.Tanh()
        elif activation == 'Sigmoid':
            act = nn.Sigmoid()

        if len(layers) == 0: lastdim = inputdim
        else: lastdim = layers[-1]

        self._init_coxph_layers(lastdim)

        blocks = [
            nn.Sequential(
                nn.Linear(self.layers[i-1] if i > 0 else inputdim, self.layers[i]),
                # nn.BatchNorm1d(self.layers[i]),
                act,
                nn.Dropout(dropout_rates[i])
            )
            for i in range(len(layers))
        ]
        
        self.embedding = nn.Sequential(*blocks)

    def forward(self, x):
        return self.expert(self.embedding(x))

class DropoutDeepCoxPH(DeepCoxPH):
    def _gen_torch_model(self, inputdim, optimizer, dropout_rates=None, activation="ReLU6"):
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)

        return DropoutDeepCoxPHTorch(inputdim, layers=self.layers, optimizer=optimizer, dropout_rates=dropout_rates, activation=activation)

    def fit(self, x, t, e, vsize=0.15, val_data=None,
          iters=1, learning_rate=1e-3, batch_size=100,
          optimizer="Adam", dropout_rates=None, activation="ReLU6"):

        r"""This method is used to train an instance of the DSM model.

        Parameters
        ----------
        x: np.ndarray
            A numpy array of the input features, \( x \).
        t: np.ndarray
            A numpy array of the event/censoring times, \( t \).
        e: np.ndarray
            A numpy array of the event/censoring indicators, \( \delta \).
            \( \delta = 1 \) means the event took place.
        vsize: float
            Amount of data to set aside as the validation set.
        val_data: tuple
            A tuple of the validation dataset. If passed vsize is ignored.
        iters: int
            The maximum number of training iterations on the training dataset.
        learning_rate: float
            The learning rate for the `Adam` optimizer.
        batch_size: int
            learning is performed on mini-batches of input data. this parameter
            specifies the size of each mini-batch.
        optimizer: str
            The choice of the gradient based optimization method. One of
            'Adam', 'RMSProp' or 'SGD'.
            
        """

        processed_data = self._preprocess_training_data(x, t, e,
                                                        vsize, val_data,
                                                        self.random_seed)

        x_train, t_train, e_train, x_val, t_val, e_val = processed_data

        #Todo: Change this somehow. The base design shouldn't depend on child

        inputdim = x_train.shape[-1]

        model = self._gen_torch_model(inputdim, optimizer, dropout_rates=dropout_rates, activation=activation)

        model, _ = train_dcph(model,
                            (x_train, t_train, e_train),
                            (x_val, t_val, e_val),
                            epochs=iters,
                            lr=learning_rate,
                            bs=batch_size,
                            return_losses=True,
                            random_seed=self.random_seed)

        self.torch_model = (model[0].eval(), model[1])
        self.fitted = True

        return self