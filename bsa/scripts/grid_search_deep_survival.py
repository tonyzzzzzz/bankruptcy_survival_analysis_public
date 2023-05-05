# %%
import os
import sys
import pandas
from copy import deepcopy
from itertools import product
from tqdm import tqdm
import numpy as np
module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)
from joblib import Parallel, delayed

# %%
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from matplotlib.axes import Axes
from auton_survival.models.cph import DeepCoxPH
from auton_survival.models.dsm import DeepSurvivalMachines
import pickle

from sklearnex import patch_sklearn
patch_sklearn()
# %%
from bsa.dataset.data_loader import load_raw_data, preprocess_data, splitting_function, drop_unknown_horizon
from bsa.utils.plots import plot_roc_curve
from bsa.models.cutomized_dcph import DropoutDeepCoxPH
# %%
import random

seed = 3407
random.seed(seed)
np.random.seed(seed)

# LOW_VAR_COL = []
LOW_VAR_COL = ['INaics3_7', 'INaics3_8', 'INaics3_12', 'INaics3_19', 'INaics3_21', 'INaics3_23', 'INaics3_78', 'INaics3_81', 'INaics3_97', 'INaics3_102']

# %%
def load_and_preprocess(path = '../data/data_for_bankruptcy_prediction_no_lags_corrected.csv', print_shape = False):
    raw_data, raw_labels = load_raw_data(path)
    x, outcomes = preprocess_data(raw_data, raw_labels)

    x_train, x_test, outcomes_train, outcomes_test = splitting_function(x, outcomes, 0.5)
    x_val, x_test, outcomes_val, outcomes_test = splitting_function(x_test, outcomes_test, 0.5)

    if print_shape:
        print("Training features shape: ", x_train.shape)
        print("Training labels shape: ", outcomes_train.shape)
        print("Testing features shape: ", x_test.shape)
        print("Testing labels shape: ", outcomes_test.shape)
        print("Validation features shape: ", x_val.shape)
        print("Validation labels shape: ", outcomes_val.shape)
    
    # Sanity check
    assert len(np.intersect1d(x_train.index.get_level_values(0).values, x_test.index.get_level_values(0).values)) == 0
    assert len(np.intersect1d(x_train.index.get_level_values(0).values, x_val.index.get_level_values(0).values)) == 0
    assert len(np.intersect1d(x_test.index.get_level_values(0).values, x_val.index.get_level_values(0).values)) == 0

    return (x_train, outcomes_train), (x_test, outcomes_test), (x_val, outcomes_val)

def get_horizon(x_train, outcomes_train, x_test, outcomes_test, x_val, outcomes_val, horizon):

    x_train = x_train.copy()
    x_test = x_test.copy()
    x_val = x_val.copy()
    outcomes_train = outcomes_train.copy()
    outcomes_test = outcomes_test.copy()
    outcomes_val = outcomes_val.copy()

    outcomes_test = drop_unknown_horizon(outcomes_test, horizon)
    outcomes_train = drop_unknown_horizon(outcomes_train, horizon)
    outcomes_val = drop_unknown_horizon(outcomes_val, horizon)

    x_train = x_train.loc[outcomes_train.index]
    x_test = x_test.loc[outcomes_test.index]
    x_val = x_val.loc[outcomes_val.index]

    # return (x_train, y_train), (x_test, y_test), (x_val, y_val)

    return (x_train, outcomes_train.values[:, 3]), (x_test, outcomes_test.values[:, 3]), (x_val, outcomes_val.values[:, 3])

# %%
def get_best_model_dcph(x_train, outcomes_train, x_val, outcomes_val, x_val_h, y_h, horizon):
    grid_params = {
        'layer_size': [16, 32, 64, 128],
        'num_layers': [1, 2, 3],
        'lr': [1e-3, 1e-4]
    }

    trial_params = [dict(zip(grid_params, v)) for v in product(*grid_params.values())]

    best_loss = 100
    best_model = None
    best_param = None

    for param in trial_params:
        ls = param['layer_size']
        n = param['num_layers']
        lr = param['lr']

        model = DeepCoxPH(layers=([ls] * n), random_seed=seed)
        model.fit(
            x=x_train,
            t=outcomes_train['T'],
            e=outcomes_train['E'],
            val_data=(x_val, outcomes_val['T'], outcomes_val['E']),
            batch_size=256,
            iters=100,
            learning_rate=lr,
        )

        y_pred_val = 1 - model.predict_survival(x_val_h, horizon)
        loss = log_loss(y_h, y_pred_val)

        if loss < best_loss:
            best_model = model
            best_loss = loss
            best_param = param
            print("Best loss: {}".format(loss))
    
    return best_model, best_loss, best_param


# %%
def get_best_model_ddcph(x_train, outcomes_train, x_val, outcomes_val, x_val_h, y_h, horizon):
    grid_params = {
        'layer_size': [16, 32, 64, 128],
        'num_layers': [1, 2, 3],
        'lr': [1e-3, 1e-4],
        'dropout': [0.0, 0.1, 0.2],
        'activation': ["ReLU6", "SeLU", "ReLU", "Tanh", "Sigmoid"]
    }

    trial_params = [dict(zip(grid_params, v)) for v in product(*grid_params.values())]

    best_loss = 100
    best_model = None
    best_param = None

    def try_param(param):
        ls = param['layer_size']
        n = param['num_layers']
        lr = param['lr']
        dropout = param['dropout']
        activation = param['activation']

        model = DropoutDeepCoxPH(layers=([ls] * n), random_seed=seed)
        model.fit(
            x=x_train,
            t=outcomes_train['T'],
            e=outcomes_train['E'],
            val_data=(x_val, outcomes_val['T'], outcomes_val['E']),
            batch_size=256,
            iters=100,
            learning_rate=lr,
            dropout_rates=[dropout] * n,
            activation=activation
        )

        y_pred_val = 1 - model.predict_survival(x_val_h, horizon)
        loss = log_loss(y_h, y_pred_val)

        return model, loss, param
    
    res = Parallel(n_jobs=-1, verbose=10)(delayed(try_param)(param) for param in trial_params)

    res = sorted(res, key=lambda x: x[1])

    return res[0][0], res[0][1], res[0][2]

# %%
def predict_auton(model, x_train, x_val, x_test, horizon):
    y_pred_train = (1 - model.predict_survival(x_train, horizon)[:, 0])
    y_pred_val = (1 - model.predict_survival(x_val, horizon)[:, 0])
    y_pred_test = (1 - model.predict_survival(x_test, horizon)[:, 0])

    return y_pred_train, y_pred_val, y_pred_test


# %%
def get_best_model_dsm(x_train, outcomes_train, x_val, outcomes_val, x_val_h, y_h, horizon):
    best_loss = 100
    best_model = None
    best_param = None

    grid_params = {
        'layer_size': [16, 32, 64, 128],
        'num_layers': [1, 2, 3],
        'lr': [1e-3, 1e-4],
        'k': [1, 3, 4, 6],
        'temp': [1, 50, 100, 1000],
        'elbo': [False, True]
    }

    trial_params = [dict(zip(grid_params, v)) for v in product(*grid_params.values())]

    def try_params(param):
        ls = param['layer_size']
        n = param['num_layers']
        lr = param['lr']
        k = param['k']
        temp = param['temp']
        elbo = param['elbo']

        model = DeepSurvivalMachines(layers=([ls] * n), random_seed=seed, k=k, temp=temp)

        model.fit(x=x_train,
            t=outcomes_train['T'],
            e=outcomes_train['E'],
            val_data=(x_val, outcomes_val['T'], outcomes_val['E']),
            batch_size=256,
            iters=100,
            learning_rate=lr,
            elbo=elbo,
        )

        y_pred_val = 1 - model.predict_survival(x_val_h, horizon)
        
        if np.isnan(y_pred_val).any():
            return None, 100, param
        
        loss = log_loss(y_h, y_pred_val)

        return model, loss, param
    
    res = Parallel(n_jobs=-1, verbose=10)(delayed(try_params)(param) for param in trial_params)

    res = sorted(res, key=lambda x: x[1])

    return res[0][0], res[0][1], res[0][2]



# %%
if __name__ == "__main__":
    figs, axs = plt.subplots(3, 3, figsize=(15, 15))
    figs_log, axs_log = plt.subplots(3, 3, figsize=(15, 15))
    
    # Load data, both in pandas format
    (x_train_raw, outcomes_train), (x_test_raw, outcomes_test), (x_val_raw, outcomes_val) = load_and_preprocess()

    for i, h in enumerate([1, 2, 5]):
            # %%
        # Load horizon, x is Dataframe, y is 1-D array for HBankruptcy
        print("Loading horizon for h={h}")
        (x_train_h, y_train), (x_test_h, y_test), (x_val_h, y_val) = get_horizon(
            x_train_raw, outcomes_train, x_test_raw, outcomes_test, x_val_raw, outcomes_val, h
        )

        # # %%
        # # MLP
        # # X, Y, cv = get_training_eval_set(x_train, y_train, x_val, y_val)
        
        # print("Training MLP...")
        # best_model_mlp = get_best_model_mlp(X, Y, cv)

        # %%
        # LR: Pass in full features and outcomes for training, data within horizon for val
        best_dcph, _ = get_best_model_dcph(x_train_raw, outcomes_train, x_val_raw, outcomes_val, x_val_h, y_val, h)


        # %%
        # Predict
        predict_dcph_train, predict_dcph_val, predict_dcph_test = predict_dcph(best_dcph, x_train_h, x_val_h, x_test_h, h)

        # %%
        # Plot ROC curve
        plot_roc_curve(y_test, [predict_dcph_test], ["DCPH"], axs[2, i], title=f"Horizon {h} - Test")
        plot_roc_curve(y_val, [predict_dcph_val], ["DCPH"], axs[1, i], title=f"Horizon {h} - Val")
        plot_roc_curve(y_train, [predict_dcph_train], ["DCPH"], axs[0, i], title=f"Horizon {h} - Train")

        plot_roc_curve(y_test, [predict_dcph_test,], ["DCPH"], axs_log[2, i], log=True, title=f"Horizon {h} - Test")
        plot_roc_curve(y_val, [predict_dcph_val,], ["DCPH"], axs_log[1, i], log=True, title=f"Horizon {h} - Val")
        plot_roc_curve(y_train, [predict_dcph_train,], ["DCPH"], axs_log[0, i], log=True, title=f"Horizon {h} - Train")

        # print("Horizon {}: RSF: {}, COX: {}".format(h, best_rsf_filename, 'experiments/best_cox.pkl'))

    figs.savefig("roc_curve.png")
    figs_log.savefig("roc_curve_log.png")