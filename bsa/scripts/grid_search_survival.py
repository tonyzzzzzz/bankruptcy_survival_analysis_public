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
from lifelines import CoxPHFitter
from pysurvival.models.survival_forest import RandomSurvivalForestModel
from pysurvival.utils import load_model
import pickle

from sklearnex import patch_sklearn
patch_sklearn()
# %%
from bsa.dataset.data_loader import load_raw_data, preprocess_data, splitting_function, drop_unknown_horizon
from bsa.utils.plots import plot_roc_curve
# %%
import random

seed = 3407
random.seed(seed)
np.random.seed(seed)

# LOW_VAR_COL = []
LOW_VAR_COL = ['INaics3_15', 'INaics3_14', 'INaics3_7', 'INaics3_8', 'INaics3_12', 'INaics3_19', 'INaics3_21', 'INaics3_23', 'INaics3_78', 'INaics3_81', 'INaics3_97', 'INaics3_102']

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

def predict_cox(model, x_train, x_val, x_test, horizon):
    x_train = x_train.copy()
    x_val = x_val.copy()
    x_test = x_test.copy()

    x_train     = x_train.drop(LOW_VAR_COL, axis=1)
    x_val       = x_val.drop(LOW_VAR_COL, axis=1)
    x_test      = x_test.drop(LOW_VAR_COL, axis=1)

    y_hat_train     = 1 - model.predict_survival_function(x_train).values[horizon, :]
    y_hat_val       = 1 - model.predict_survival_function(x_val).values[horizon, :]
    y_hat_test      = 1 - model.predict_survival_function(x_test).values[horizon, :]

    return y_hat_train, y_hat_val, y_hat_test

def get_best_model_cox(x_train, outcomes_train, x_val, y_val, horizon):

    best_alpha = None
    best_loss = 100
    best_mod_cox = None
 
    data = x_train.join(outcomes_train)
    data = data.copy()

    data = data.drop('IBankrupt', axis=1)
    data = data.drop(LOW_VAR_COL, axis=1)

    x_val = x_val.copy()
    x_val = x_val.drop(LOW_VAR_COL, axis=1)

    for alpha in [1, 0.1, 0.01, 0.001]:

        model = CoxPHFitter(penalizer=alpha).fit(data, duration_col='T', event_col='E')
        y_pred_val = 1 - model.predict_survival_function(x_val).values[horizon, :] # Subtract the probability from 1 to get probability of bankruptcy

        loss = log_loss(y_val, y_pred_val)

        if loss < best_loss:
            best_alpha = alpha
            best_loss = loss
            best_mod_cox = model
            print("Best Cox Loss: {}, penalizer: {}".format(loss, alpha))

    return best_mod_cox, best_alpha, best_loss

# %%
def predict_rsf(model_path, x_train, x_val, x_test, horizon):

    model = load_model(model_path)

    y_pred_train = 1 - model.predict_survival(x_train, horizon)
    y_pred_val = 1 - model.predict_survival(x_val, horizon)
    y_test_val = 1 - model.predict_survival(x_test, horizon)

    return y_pred_train, y_pred_val, y_test_val

# %%
def get_best_model_rsf(x_train, outcomes_train, x_val, y_val, horizon):
    best_loss = 100
    best_mod_rf = None

    grid_params = {
        'n_est': [50, 100, 200],
        'max_feat': ["sqrt", 50, 75],
        'm_depth': [5, 6, 8, 10]
    }

    trial_params = [dict(zip(grid_params, v)) for v in product(*grid_params.values())]

    num_threads_per_job = 4
    num_jobs = 112 // 4

    def try_param(param):
        d, e, f = param['m_depth'], param['n_est'], param['max_feat']
        rsf = RandomSurvivalForestModel(num_trees=e)
        rsf.fit(x_train, outcomes_train['T'].values, outcomes_train['E'].values, max_features=f, max_depth=d, sample_size_pct=1.0, num_threads=num_threads_per_job)
        
        filename = "experiments/rsf-{}-{}-{}-{}.zip".format(horizon, d, e, f)
        rsf.save(filename)

        y_pred_val = 1 - rsf.predict_survival(x_val, horizon)
        loss = log_loss(y_val, y_pred_val)

        return filename, loss

    res = Parallel(n_jobs=num_jobs, verbose=10)(delayed(try_param)(param) for param in trial_params)

    res = sorted(res, key=lambda x: x[1])

    # for max_depth in [5, 6, 8, 10]:
    #     for n_estimators in [50, 100, 200]:
    #         for max_features in ["sqrt", 50, 75]:
    #             print("Trying parameters: {}, {}, {}".format(max_depth, n_estimators, max_features))
    #             rsf = RandomSurvivalForestModel(num_trees=n_estimators)
    #             rsf.fit(x_train, outcomes_train['T'].values, outcomes_train['E'].values, max_features=max_features, max_depth=max_depth, sample_size_pct=1.0, num_threads=-1)

    #             y_pred_val = 1 - rsf.predict_survival(x_val, horizon)
    #             loss = log_loss(y_val, y_pred_val)

    #             if loss < best_loss:
    #                 best_loss = loss
    #                 best_mod_rf = rsf
    #                 print("Best RSF Loss: {}, Depth: {}, N_Estimators: {}, Max_feats: {}".format(loss, max_depth, n_estimators, max_features))

    return res[0][0], res[0][1]
# %%
if __name__ == "__main__":
    figs, axs = plt.subplots(3, 3, figsize=(15, 15))
    figs_log, axs_log = plt.subplots(3, 3, figsize=(15, 15))
    
    # Load data, both in pandas format
    (x_train_raw, outcomes_train), (x_test_raw, outcomes_test), (x_val_raw, outcomes_val) = load_and_preprocess()

    with open('experiments/x_train_raw.pkl', 'wb') as f:
        pickle.dump(x_train_raw, f)
    
    with open('experiments/outcomes_train', 'wb') as f:
        pickle.dump(outcomes_train, f)

    with open('experiments/x_test_raw.pkl', 'wb') as f:
        pickle.dump(x_test_raw, f)

    with open('experiments/outcomes_test.pkl', 'wb') as f:
        pickle.dump(outcomes_test, f)

    with open('experiments/x_val_raw.pkl', 'wb') as f:
        pickle.dump(x_val_raw, f)

    with open('experiments/outcomes_val', 'wb') as f:
        pickle.dump(outcomes_val, f)

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
        best_cox, _, _ = get_best_model_cox(x_train_raw, outcomes_train, x_val_h, y_val, horizon=h)
        best_rsf_filename, _ = get_best_model_rsf(x_train_raw, outcomes_train, x_val_h, y_val, horizon=h)

        # Save COX model
        with open('experiments/best_cox.pkl', 'wb') as f:
            pickle.dump(best_cox, f)

        # %%
        # Predict
        # predict_mlp_train, predict_mlp_val, predict_mlp_test = predict_mlp(best_model_mlp, x_train, x_val, x_test)
        predict_cox_train, predict_cox_val, predict_cox_test = predict_cox(best_cox, x_train_h, x_val_h, x_test_h, h)
        predict_rsf_train, predict_rsf_val, predict_rsf_test = predict_rsf(best_rsf_filename, x_train_h, x_val_h, x_test_h, h)

        # %%
        # Plot ROC curve
        plot_roc_curve(y_test, [predict_cox_test, predict_rsf_test], ["Cox", "RSF"], axs[2, i], title=f"Horizon {h} - Test")
        plot_roc_curve(y_val, [predict_cox_val, predict_rsf_val], ["Cox", "RSF"], axs[1, i], title=f"Horizon {h} - Val")
        plot_roc_curve(y_train, [predict_cox_train, predict_rsf_train], ["Cox", "RSF"], axs[0, i], title=f"Horizon {h} - Train")

        plot_roc_curve(y_test, [predict_cox_test, predict_rsf_test], ["Cox", "RSF"], axs_log[2, i], log=True, title=f"Horizon {h} - Test")
        plot_roc_curve(y_val, [predict_cox_val, predict_rsf_val], ["Cox", "RSF"], axs_log[1, i], log=True, title=f"Horizon {h} - Val")
        plot_roc_curve(y_train, [predict_cox_train, predict_rsf_train], ["Cox", "RSF"], axs_log[0, i], log=True, title=f"Horizon {h} - Train")

        print("Horizon {}: RSF: {}, COX: {}".format(h, best_rsf_filename, 'experiments/best_cox.pkl'))

    figs.savefig("roc_curve.png")
    figs_log.savefig("roc_curve_log.png")