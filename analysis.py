# %%
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# TensorFlow imports
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.utils import Sequence, to_categorical

from tensorflow.keras.callbacks import TensorBoard

from models.losses import Losses
from models.model import GMM_Survival
from utils.eval_utils import (accuracy_metric, calibration, cindex,
                              cindex_metric)
from utils.utils import ProgBar
import warnings
tfd = tfp.distributions
tfkl = tf.keras.layers
tfpl = tfp.layers
tfk = tf.keras
# VaDeSC model

# Set Seaborn style for a cleaner look
sns.set(style="whitegrid")

print(os.getcwd())
warnings.filterwarnings("ignore")


# VaDeSC model


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'


def print_layer_details(layer, indent=0):
    indent_str = ' ' * indent
    print(f"{indent_str}Layer: {layer.name}")

    # Check if this layer contains sublayers
    if hasattr(layer, 'layers') and layer.layers:
        # If so, recursively print sublayer details
        for sublayer in layer.layers:
            print_layer_details(sublayer, indent + 2)
    else:
        # If not, it's a leaf node, print its variables
        if hasattr(layer, 'variables'):
            for var in layer.variables:
                print(f"{indent_str}  Variable: {var.name}, Shape: {var.shape}")
    # For layers without variables directly accessible, this part will be skipped

# Check for available GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("Using GPU:", gpus[0].name)
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
else:
    print("No GPU found. Using CPU.")

# %% [markdown]
# ## Dataset

# %%
# from src.radcure import Radcure

def del_multicol(df):
    return df.drop(columns=df.filter(regex='^N Stage_N[1-9X]').columns).drop(columns=["T Stage_T3/4", "Stage_III/IV","ECOG_0"]) \
        .drop(columns=df.filter(regex='^Disease').columns).drop(columns="target_binary") \
        .drop(columns=["Sex_Female"])

# df_tr = Radcure.make_data("../data/radcure/clinical_train.csv")
# df_te = Radcure.make_data("/u/97/nguyenq10/unix/HUS/simpy-vadesc/data/radcure/clinical_test.csv")
df = pd.read_csv("./datasets/radcure/clinical_train.csv").drop(columns=["split","target_binary","Study ID"])
df = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)
df.columns = [col.lower().strip() for col in df.columns]

cat_feats = ["sex", "disease site", "t stage", "n stage", "hpv combined", "chemo?", "ecog", "stage"]
num_feats = ["age"]
outcomes = df[["survival_time","death"]].rename(columns={"death":"event","survival_time":"time"})
features = df.drop(columns=["survival_time","death"])

from auton_survival.preprocessing import Preprocessor

preprocessor = Preprocessor(cat_feat_strat='ignore', num_feat_strat= 'mean') 
x = preprocessor.fit_transform(features, cat_feats=cat_feats, num_feats=num_feats,
                                one_hot=True, fill_value=-1)
y = outcomes


from sklearn.model_selection import train_test_split


x_tr, x_val, y_tr, y_val = train_test_split(x, y, test_size=0.3, random_state=1) 
dataset = "radcure"
# Define the times for tuning the model hyperparameters and for evaluating the model
times = np.quantile(y_tr['time'][y_tr['event']==1], np.linspace(0.1, 1, 10)).tolist()
print(f'Number of training data points: {len(x_tr)}')
print(f'Number of validation data points: {len(x_val)}')

# %% [markdown]
# ## VaDeSC Model

# %%
class DataGen(Sequence):
    def __init__(self, X, y):
        self.X = X.values
        self.y = y.values

    def __getitem__(self, index):
        return (self.X, self.y), {"output_1": self.X, "output_5": self.y}

    def __len__(self):
        return 1
    
# Assuming x_tr is your numpy array of training data
gen_tr = DataGen(x_tr, y_tr)
gen_val = DataGen(x_val, y_val)
gen = DataGen(x,y)

(next(iter(gen_tr))[0][0].shape,next(iter(gen_tr))[0][1].shape, )

# %%
"Model"
from copy import deepcopy
from utils.utils import get_workdir

import torch
from auton_survival.metrics import phenotype_purity, survival_regression_metric
from lifelines.utils import concordance_index
from sklearn.model_selection import ParameterGrid

param_grid = {
    'num_clusters': [2],
    'learning_rate': [1e-3],
    "seed": [0],
    "layers": [[50, 100]]
}

params = ParameterGrid(param_grid)


# Define the times for tuning the model hyperparameters and for evaluating the model
# times = np.quantile(y_tr['time'][y_tr['event']==1], np.linspace(0.1, 1, 10)).tolist()
# Perform hyperparameter tuning
models = []
base_cfg = {"inp_shape": 45, "num_clusters": 2, "latent_dim": 8, 'monte_carlo': 1,
            'learn_prior': False, 'weibull_shape': 2.5, 'sample_surv': False,
            "activation": None, "survival": True, "epochs": 1000, "layers": [50, 100],
            "seed": 0, "learning_rate": 1e-3}
# Define reconstruction loss function
loss = Losses(base_cfg).loss_reconstruction_mse

cis = []
for param in params:
    cfg = deepcopy(base_cfg)
    for k, v in param.items():
        if k in cfg:
            cfg[k] = v
    print(cfg)
    # Construct the model & optimizer
    setup_seed(cfg["seed"])
    progress = ProgBar(epochs=cfg["epochs"], metrics=[
                       "loss", "val_output_5_cindex_metric"])
    model = GMM_Survival(**cfg)
    tb_cb = TensorBoard(
        log_dir=get_workdir(dataset), write_graph=True)
    tf.keras.backend.set_value(model.use_t, np.array([1.0]))
    optimizer = tf.keras.optimizers.legacy.Adam(
        learning_rate=cfg["learning_rate"], decay=0.00001)
    model.compile(optimizer, loss={"output_1": loss}, run_eagerly=False,
                  metrics={"output_5": cindex_metric})
    history = model.fit(gen, validation_data=gen_val,
                        epochs=cfg["epochs"], verbose=0,
                        callbacks=[progress, tb_cb])
    print("loss", history.history['loss'][-1])
    rec, z_sample, p_z_c, p_c_z, risks, lambdas = model.predict(
        (x_val.values, y_val.values))
    risks = np.squeeze(risks)
    phenotypes = np.argmax(p_c_z, -1)
    metric_val1 = phenotype_purity(phenotypes_train=phenotypes, outcomes_train=y_val,
                                   phenotypes_test=None, outcomes_test=None,
                                   strategy='integrated', horizons=[6],
                                   bootstrap=None)
    # metric_val1 = survival_regression_metric(
    #     'ibs', y_val, predictions_val, times, y_tr)
    # metric_val2 = concordance_index(y_val.time, risk, y_val.event)

    models.append([metric_val1, model, cfg])
    ci = cindex(t=y_val.values[:, 0], e=y_val.values[:, 1], scores_pred=risks)
    print("ci", ci)
    cis.append(ci)


print(f"Concordance: {np.mean(cis)}(std. {np.std(cis)})")

# Select the best model based on the mean metric value computed for the validation set
models = sorted(models, key=lambda x: x[0], reverse=True)
best_model = models[0][1]
print(models)
print_layer_details(best_model)

# %%
from auton_survival import reporting
from auton_survival.metrics import phenotype_purity

# Estimate the probability of event-free survival for phenotypes using the Kaplan Meier estimator.
rec, z_sample, p_z_c, p_c_z, risk_scores, lambdas = best_model.predict((x_val.values, y_val.values), verbose=0, batch_size=len(x_val))
# phenotypes = np.argmax(p_c_z, -1)
# phenotypes = np.random.randint(0,1,len(x_val))
reporting.plot_kaplanmeier(y_val, groups=phenotypes)

plt.xlabel('Time (Years)')
plt.ylabel('Event-Free Survival Probability')
plt.legend(['Phenotype 1', 'Phenotype 2'], loc="upper right")
plt.show()

# %% [markdown]
# ## Cluster Analysis

# %%
# - Phase 1: Dimensionality reduction and clustering
# - Define a VAE model with 40 intermediate units and 2 latent units
# VAE = create_VAE_model(40, 2)

# - Train the VAE model on the dataset of N patients with a specific index disease for 30,000 epochs
# VAE.train(dataset, epochs = 30000)

# - Initialize an empty list to store the cluster labels of the subsampled datasets
# cluster_labels = []

# - Repeat 100 times
# for i in range(100):
#   - Randomly subsample N/2 patients from the dataset
#   subsample = random_sample(dataset, size = N/2)

#   - Encode the subsample into the latent space of the VAE model
#   latent_vectors = VAE.encode(subsample)

#   - Cluster the latent vectors using HDBSCAN with min_cluster_size = N/100 and min_samples = 5
#   Cruns = HDBSCAN(latent_vectors, min_cluster_size = N/100, min_samples = 5)

# param_grid = {
#     "seed": [0, 1],
#     "num_clusters": [2,3,4],
# }
param_grid = {
    "seed": range(100),
}

params = ParameterGrid(param_grid)

#   - Append the cluster labels to the cluster_labels list
#   cluster_labels.append(Cruns.labels)
cfg = {"inp_shape": 45, "num_clusters": 2, "latent_dim": 8, 'monte_carlo': 1,
       'learn_prior': False, 'weibull_shape': 2.5, 'sample_surv': False,
       "activation": None, "survival": True, "epochs": 1000, "layers": [50, 100],
       "seed": 0, "learning_rate": 1e-3}
N = len(x)
K = cfg["num_clusters"]
runs = len(params)
Nsub = int(N*0.7)
models, Cruns, metric_vals = [],[],[]
x_indices = []
for param in params:
    cfg = deepcopy(base_cfg)
    for k, v in param.items():
        if k in cfg:
            cfg[k] = v
    print(cfg)
    setup_seed(cfg["seed"])

    x_tr1, x_val1, y_tr1, y_val1 = train_test_split(
        x, y, train_size=Nsub, random_state=cfg["seed"])
    # Assuming x_tr is your numpy array of training data
    gen_tr1 = DataGen(x_tr1, y_tr1)
    gen_val1 = DataGen(x_val1, y_val1)
    x_indices.append(list(x_tr1.index))
    # Construct the model & optimizer

    progress = ProgBar(epochs=cfg["epochs"], metrics=[
                       "loss", "val_output_5_cindex_metric"])
    model = GMM_Survival(**cfg)
    # tb_cb = TensorBoard(
    #     log_dir=get_workdir(dataset), write_graph=True)
    tf.keras.backend.set_value(model.use_t, np.array([1.0]))
    optimizer = tf.keras.optimizers.legacy.Adam(
        learning_rate=cfg["learning_rate"], decay=0.00001)
    model.compile(optimizer, loss={"output_1": loss}, run_eagerly=False,
                  metrics={"output_5": cindex_metric})
    history = model.fit(gen_tr1, validation_data=gen_val1,
                        epochs=cfg["epochs"], verbose=0,
                        callbacks=[progress, tb_cb])
    rec, z_sample, p_z_c, p_c_z, risks, lambdas = model.predict(
        (x_val1.values, y_val1.values))
    phenotypes = np.argmax(p_c_z, -1)
    Cruns.append(phenotypes)
    metric_val1 = phenotype_purity(phenotypes_train=phenotypes, outcomes_train=y_val1,
                                   phenotypes_test=None, outcomes_test=None,
                                   strategy='integrated', horizons=[6],
                                   bootstrap=None)
    models.append(model)
    print(metric_val1)


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
M

# %%
rec, z_sample, p_z_c, p_c_z, risks, lambdas = best_model.predict(
    (x.values, y.values),batch_size=len(x))
Cmain = np.argmax(p_c_z, -1)

def consensus_idx(Cmain, k, M=M):
    Ck = np.where(Cmain == k)[0]
    if len(Ck) > 1:
        # Select the relevant sub-matrix for cluster k
        Mk = M[np.ix_(Ck, Ck)]
    
        # Sum above the diagonal elements to get consensus sum for unique pairs
        consensus_sum = np.sum(np.triu(Mk,1))
    
        # Number of unique pairs in cluster k is "N choose 2"
        num_pairs = len(Ck) * (len(Ck) - 1) / 2
    
        # Calculate the consensus index for cluster k
        return consensus_sum / num_pairs
    else:
        # Handle the case where a cluster may have only one element
        return 0

m = []
for k in range(K):
    m.append(consensus_idx(Cmain,k))


print(f"**consensus_indices {m}**")

# %%
# 1. H0: the consensus index is the same regardless of cluster assignments
# 2. test statistics is consensus index
# distribution 
#


# %%
P = 10000
p_values = []
for k in range(K):
    mk_perms = [] 
    for p in range(P):
        Cmain_p = np.random.permutation(Cmain)
        mkp = consensus_idx(Cmain_p,k)
        mk_perms.append(mkp)
    # Calculate the p-value for the observed consensus index
    p_value = (np.sum(mk_perms >= consensus_idx(Cmain,k)) + 1) / (P + 1)  # Adding 1 to the numerator and denominator for continuity correction
    p_values.append(p_value)
    

print(f"**p_values {p_values}**")

