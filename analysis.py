# Import necessary libraries
import os
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from copy import deepcopy
from sklearn.model_selection import ParameterGrid, train_test_split
from tensorflow.keras.callbacks import TensorBoard

# Custom module imports
from datasets.radcure import load_radcure
from main import train
from models.losses import Losses
from models.model import GMM_Survival
from utils.eval_utils import accuracy_metric, calibration, cindex, cindex_metric
from utils.utils import ProgBar, get_latent, get_purity, get_workdir, setup_env, setup_seed
import sys

# TensorFlow Probability shortcuts
tfd = tfp.distributions
tfkl = tf.keras.layers
tfpl = tfp.layers
tfk = tf.keras

# Environment setup for reproducibility and current directory printout
setup_env()
print(os.getcwd())

# Load dataset
path = "/u/97/nguyenq10/unix/HUS/cancer-clusters/datasets/radcure/clinical1.csv"
x, y, cfg0 = load_radcure(path)
x_tr, x_val, y_tr, y_val = train_test_split(
    x, y, train_size=cfg0["train_size"], random_state=cfg0["seed"])

# Configuration for the dataset and model
if len(sys.argv) > 1:
    cfg0["num_clusters"] = int(sys.argv[1])

workdir0 = "radcure/analysis"
workdir = get_workdir(workdir0, f"k{cfg0['num_clusters']}")
cfg0["workdir"] = workdir


# Training the model with specified configuration
print("Training main model...")
model0, _ = train(x, y, config=cfg0, validation_data=(x_val, y_val))
C0 = model0.get_phenotypes(x, y)
print(get_purity(y_val, model0.get_phenotypes(x_val, y_val)))
get_latent(model0, x, y)


# Cluster Analysis
def train_shuffled(x, y, params, base_cfg, Nsub=0.5):
    base_cfg["train_size"] = Nsub
    models, Cruns, x_indices = [], [], []
    for i, param in enumerate(params):
        print(f"subsampled model {i}/{len(params)}")
        cfg = {k: param[k] if k in param else v for k, v in base_cfg.items()}
        cfg["workdir"] = cfg["workdir"] + f"/seed_{cfg['seed']}"
        print(cfg)
        x_tr1, x_val1, y_tr1, y_val1 = train_test_split(
            x, y, train_size=cfg["train_size"], random_state=cfg["seed"])
        model,_ = train(x_tr1, y_tr1, validation_data=(
            x_val1, y_val1), config=cfg, log=False)
        x_indices.append(list(x_tr1.index))
        Cruns.append(model.get_phenotypes(x_tr1, y_tr1))
        phe_val1 = model.get_phenotypes(x_val1, y_val1)
        purity = get_purity(y_val1, phe_val1)
        print(f"purity: {purity}")
        models.append([model, cfg, purity])
    return Cruns, x_indices, models


print("Training subsampled models...")
param_grid = {"seed": range(100)}
params = ParameterGrid(param_grid)
Cruns, x_indices, models = train_shuffled(x, y, params, cfg0)

# Consensus clustering matrix calculation
N = len(x)
runs = len(params)
I = np.zeros((runs, N, N))
M0 = np.zeros((runs, N, N))

for run in range(runs):
    idx = np.array(x_indices[run])
    mask = np.zeros(N, dtype=bool)
    mask[idx] = True
    mask_mat = np.outer(mask, mask)
    I[run] = mask_mat.astype(int)

    labels = np.array(Cruns[run])
    label_mat = np.equal.outer(labels, labels)
    idx_mat = np.ix_(idx, idx)
    M0[run][idx_mat] = label_mat


M = np.nan_to_num(np.sum(M0, axis=0) / np.sum(I, axis=0))
print(M)


# Consensus index calculation
def consensus_idx(C0, k, M=M):
    Ck = np.where(C0 == k)[0]
    if len(Ck) > 1:
        Mk = M[np.ix_(Ck, Ck)]
        consensus_sum = np.sum(np.triu(Mk, 1))
        num_pairs = len(Ck) * (len(Ck) - 1) / 2
        return consensus_sum / num_pairs
    else:
        return 0


m = [consensus_idx(C0, k) for k in range(cfg0["num_clusters"])]
# P-values calculation
P = 10000
p_values = []
for k in range(cfg0["num_clusters"]):
    mk_perms = np.array(
        [consensus_idx(np.random.permutation(C0), k) for p in range(P)])
    p_value = (np.sum(mk_perms >= consensus_idx(C0, k)) + 1) / (P + 1)
    p_values.append(p_value)

# Save results to files
pd.DataFrame(M).to_csv(f"{workdir}/cm.csv", index=False)
print(f"Consensus matrix saved to {workdir}/cm.csv")
pd.DataFrame(m, columns=['CI']).to_csv(f"{workdir}/ci.csv", index=False)
print(f"Consensus indices saved to {workdir}/ci.csv")
pd.DataFrame(p_values, columns=['PV']).to_csv(f"{workdir}/pv.csv", index=False)
print(f"P-values saved to {workdir}/pv.csv")
