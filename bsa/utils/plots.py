import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score

def plot_roc_curve(y, predictions, labels, ax, log=False, title=""):

    for i, p in enumerate(predictions):
        fpr, tpr, _ = roc_curve(y, p)
        auc = roc_auc_score(y, p)
        
        ax.plot(fpr, tpr, label="{} AUC={}".format(labels[i], auc))

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