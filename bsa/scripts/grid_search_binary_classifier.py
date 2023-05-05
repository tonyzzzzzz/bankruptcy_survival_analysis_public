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
from torch.nn import modules
from torch import nn
import torch
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from skorch import NeuralNet
from matplotlib.axes import Axes
import skorch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
print("Device: ", device)

# %%
from bsa.dataset.data_loader import load_raw_data, preprocess_data, splitting_function, drop_unknown_horizon
from bsa.models.mlp import MLP, CheckpointCallback
# %%
import random

seed = 3407
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
# torch.use_deterministic_algorithms(True)

if device == 'cuda':
    torch.backends.cudnn.deterministic = True

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

    return (x_train.values, outcomes_train.values[:, 3]), (x_test.values, outcomes_test.values[:, 3]), (x_val.values, outcomes_val.values[:, 3])
# %%
def get_training_eval_set(x_train, y_train, x_val, y_val):
    cv = [(np.arange(0, x_train.shape[0]), np.arange(x_train.shape[0], x_train.shape[0] + x_val.shape[0]))]
    X = torch.cat((torch.from_numpy(x_train).float(), torch.from_numpy(x_val).float()), dim=0)
    Y = torch.cat((torch.from_numpy(y_train).float().unsqueeze(1), torch.from_numpy(y_val).float().unsqueeze(1)), dim=0)

    return X, Y, cv

# %%
def get_best_model_mlp(X, Y, cv):


    grid_params = {
        'module__layer_size': [16, 32, 64, 128],
        'module__num_layers': [1, 2, 3],
        'module__dropout_rate': [0.0, 0.1, 0.2],
        'module__activation': [nn.ReLU, nn.LeakyReLU, nn.GELU, nn.Tanh, nn.Sigmoid, nn.SELU],
        'max_epochs': [100],
        'batch_size': [256]
    }

    trial_params = [dict(zip(grid_params, v)) for v in product(*grid_params.values())]

    best_model = None
    best_params = None
    best_epoch = None

    def try_param(param):
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        
        skorch_net = NeuralNet(
            MLP,
            module__input_feature=X.shape[1],
            module__layer_size=64,
            module__num_layers=1,
            module__activation=nn.Sigmoid,
            module__dropout_rate=0.2,
            criterion=nn.BCEWithLogitsLoss,
            optimizer=torch.optim.AdamW,
            optimizer__lr=0.001,
            optimizer__weight_decay=0.01,
            max_epochs=100,
            batch_size=256,
            device='cpu',
            train_split=skorch.dataset.ValidSplit(cv=cv),
            verbose=0,
            callbacks=[
                ('early_stopping', skorch.callbacks.EarlyStopping('valid_loss')),
                ('checkpoint', CheckpointCallback())
            ],
        )
        skorch_net.set_params(**param)
        skorch_net.fit(X, Y)
        history = skorch_net.history
        if(len(history)) < 100:
            cur_loss = skorch_net.history[-6]['valid_loss']
            epoch = len(history) - 5
            # Revert weights
            skorch_net.module_.load_state_dict(skorch_net.history[-6]['checkpoint'])
        else:
            cur_loss = skorch_net.history[-1]['valid_loss']
            epoch = len(history)
        
        m = deepcopy(skorch_net.module_)

        return m, cur_loss, epoch, param
    
    res = Parallel(n_jobs=-1, verbose=10)(delayed(try_param)(param) for param in trial_params)

    res = sorted(res, key=lambda x: x[1])

    return res[0][0], res[0][2], res[0][3]
# %%
def predict_mlp(model, x_train, x_val, x_test):
    model.eval()
    y_hat_train = model.forward(torch.from_numpy(x_train).float().to(device))[:, 0].cpu().detach().numpy()
    y_hat_val = model.forward(torch.from_numpy(x_val).float().to(device))[:, 0].cpu().detach().numpy()
    y_hat_test = model.forward(torch.from_numpy(x_test).float().to(device))[:, 0].cpu().detach().numpy()

    return y_hat_train, y_hat_val, y_hat_test 


# %%
def get_best_model_lr(x_train, y_train, x_val, y_val):
    best_alpha = None
    best_loss = 100
    best_mod_lr = None

    for alpha in [1, 0.1, 0.001, 0.0001, 0.00001]:
        clf = SGDClassifier(max_iter=10000, random_state=seed, loss='log', n_jobs=-1, alpha=alpha)
        clf.fit(x_train, y_train) 
        y_pred_val = clf.predict_proba(x_val)[:, 1]
        loss = log_loss(y_val, y_pred_val)
        if loss < best_loss:
            best_alpha = alpha
            best_loss = loss
            best_mod_lr = clf
            print("loss: {}".format(loss))

    return best_mod_lr, best_alpha, best_loss

# %%
def get_best_model_rf(x_train, y_train, x_val, y_val):
    best_loss = 100
    best_mod_rf = None
    best_params = None

    for max_depth in [5, 6, 8, 10]:
        for n_estimators in [50, 100, 200]:
            for max_features in [None, "sqrt", 50, 75]:
                rf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, max_depth=max_depth, max_features=max_features, random_state=seed)
                rf.fit(x_train, y_train)
                y_pred_val = rf.predict_proba(x_val)[:, 1]
                loss = log_loss(y_val, y_pred_val)
                if loss < best_loss:
                    best_loss = loss
                    best_mod_rf = rf
                    best_params = (max_depth, n_estimators, max_features)
                    print("loss: {}".format(loss))

    return best_mod_rf, best_loss, best_params
# %%
def predict_sk(model, x_train, x_val, x_test):
    y_hat_train = model.predict_proba(x_train)[:, 1]
    y_hat_val = model.predict_proba(x_val)[:, 1]
    y_hat_test = model.predict_proba(x_test)[:, 1]

    return y_hat_train, y_hat_val, y_hat_test
# %%
def plot_roc_curve(y, predict_mlp, predict_rf, predict_lr, ax : Axes, log=False, title=""):

    fpr_mlp, tpr_mlp, thresholds = roc_curve(y, predict_mlp)
    fpr_rf, tpr_rf, _ = roc_curve(y, predict_rf)
    fpr_lr, tpr_lr, _ = roc_curve(y, predict_lr)

    mlp_auc = roc_auc_score(y, predict_mlp)
    rf_auc = roc_auc_score(y, predict_rf)
    lr_auc = roc_auc_score(y, predict_lr)

    ax.plot(fpr_mlp, tpr_mlp, label=f'MLP AUC:{mlp_auc:.3f}')
    ax.plot(fpr_rf, tpr_rf, label=f'RF AUC:{rf_auc:.3f}')
    ax.plot(fpr_lr, tpr_lr, label=f'LR AUC:{lr_auc:.3f}')

    if log:
        ax.plot(np.linspace(0, 1, 10000), np.linspace(0, 1, 10000), linestyle='--', label='Random', alpha=0.5)
    else:
        ax.plot([0, 1], [0, 1], linestyle='--', label='Random', alpha=0.5)
    
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend()
    if log:
        ax.set_xscale('log')

# %%
if __name__ == "__main__":
    figs, axs = plt.subplots(3, 3, figsize=(15, 15))
    figs_log, axs_log = plt.subplots(3, 3, figsize=(15, 15))
    
    (x_train_raw, y_train_raw), (x_test_raw, y_test_raw), (x_val_raw, y_val_raw) = load_and_preprocess()

    for i, h in enumerate([1, 2, 5]):
            # %%
        # Load data
        print("Loading horizon for h={h}")
        (x_train, y_train), (x_test, y_test), (x_val, y_val) = get_horizon(x_train_raw, y_train_raw, x_test_raw, y_test_raw, x_val_raw, y_val_raw, h)

        # %%
        # MLP
        X, Y, cv = get_training_eval_set(x_train, y_train, x_val, y_val)
        
        print("Training MLP...")
        best_model_mlp = get_best_model_mlp(X, Y, cv)

        # %%
        # LR
        best_lr, _, _ = get_best_model_lr(x_train, y_train, x_val, y_val)
        best_rf, _ = get_best_model_rf(x_train, y_train, x_val, y_val)

        # %%
        # Predict
        predict_mlp_train, predict_mlp_val, predict_mlp_test = predict_mlp(best_model_mlp, x_train, x_val, x_test)
        predict_lr_train, predict_lr_val, predict_lr_test = predict_sk(best_lr, x_train, x_val, x_test)
        predict_rf_train, predict_rf_val, predict_rf_test = predict_sk(best_rf, x_train, x_val, x_test)

        # %%
        # Plot ROC curve
        
        plot_roc_curve(y_test, predict_mlp_test, predict_rf_test, predict_lr_test, axs[2, i], title=f"Horizon {h} - Test")
        plot_roc_curve(y_val, predict_mlp_val, predict_rf_val, predict_lr_val, axs[1, i], title=f"Horizon {h} - Val")
        plot_roc_curve(y_train, predict_mlp_train, predict_rf_train, predict_lr_train, axs[0, i], title=f"Horizon {h} - Train")

        plot_roc_curve(y_test, predict_mlp_test, predict_rf_test, predict_lr_test, axs_log[2, i], log=True, title=f"Horizon {h} - Test")
        plot_roc_curve(y_val, predict_mlp_val, predict_rf_val, predict_lr_val, axs_log[1, i], log=True, title=f"Horizon {h} - Val")
        plot_roc_curve(y_train, predict_mlp_train, predict_rf_train, predict_lr_train, axs_log[0, i], log=True, title=f"Horizon {h} - Train")

    figs.savefig("roc_curve.png")
    figs_log.savefig("roc_curve_log.png")
