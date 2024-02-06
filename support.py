# %%
import os
import random
import sys
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import yaml
from auton_survival.metrics import survival_regression_metric
from pandas import DataFrame
from sklearn.metrics.cluster import (adjusted_rand_score,
                                     normalized_mutual_info_score)
from sklearn.model_selection import ParameterGrid
from tensorflow.keras.utils import Sequence, to_categorical

import utils.utils as utils
from datasets.support import generate_support
from models.losses import Losses
from models.model import GMM_Survival
from utils.eval_utils import (accuracy_metric, calibration, cindex,
                              cindex_metric)
from utils.plotting import (plot_bigroup_kaplan_meier, plot_group_kaplan_meier,
                            plot_tsne_by_cluster, plot_tsne_by_survival)
from utils.utils import ProgBar

# sys.path.insert(0, '../')
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# TensorFlow imports
tfd = tfp.distributions
tfkl = tf.keras.layers
tfpl = tfp.layers
tfk = tf.keras

# VaDeSC model


# %%
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'


# Check for available GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("Using GPU:", gpus[0].name)
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
else:
    print("No GPU found. Using CPU.")
# %%
# Fix random seed

# %% [markdown]
# ### SUPPORT Data


# %%
# Generate survival MNIST data

x_tr, x_val, x_te, t_tr, t_val, t_te, e_tr, e_val, e_te, c_tr, c_val, c_te = generate_support(
    seed=42)
# y_tr = DataFrame.from_dict({"time": t_tr, "event": e_tr})
# y_val = DataFrame.from_dict({"time": t_val, "event": e_val})
# Wrap t, d, and c together
y_tr = np.stack([t_tr, e_tr, c_tr], axis=1)
y_val = np.stack([t_val, e_val, c_val], axis=1)
# Define the times for tuning the model hyperparameters and for evaluating the model
times = np.quantile(t_val[e_val == 1],
                    np.linspace(0.1, 1, 10)).tolist()

# Visualise t-SNE of images colour data points according to the cluster
# plot_tsne_by_cluster(X=x_tr, c=c_tr, font_size=12, seed=42)
x_tr.shape

# %% [markdown]
# ### VaDeSC Model

# %%

# # Load config file for the MNIST
# project_dir = os.path.dirname(os.getcwd())
# config_path = Path("/u/97/nguyenq10/unix/HUS/vadesc_/configs/mnist.yml")
# with config_path.open(mode='r') as yamlfile:
#     configs = yaml.safe_load(yamlfile)
# print(configs)


class DataGen(Sequence):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, index):
        return (self.X, self.y), {"output_1": self.X, "output_5": self.y}

    def __len__(self):
        return 1


# tf.config.run_functions_eagerly(True)
# Hyperparameters
learning_rate = 0.001
num_epochs = 1000


# Create a tf.data.Dataset
train_data = tf.data.Dataset.from_tensor_slices((x_tr, t_tr, e_tr))
# Assuming x_tr is your numpy array of training data
gen_tr = DataGen(x_tr, y_tr)
gen_val = DataGen(x_val, y_val)


# %%

# Define parameters for tuning the model
param_grid = {'num_clusters': [3],
              'learning_rate': [1e-3],
              "seed": [0]
              }

params = ParameterGrid(param_grid)

# Define the times for tuning the model hyperparameters and for evaluating the model
# times = np.quantile(y_tr['time'][y_tr['event']==1], np.linspace(0.1, 1, 10)).tolist()
# Perform hyperparameter tuning
models = []
base_cfg = {"inp_shape": 59, "num_clusters": 2, "latent_dim": 16, 'monte_carlo': 1,
            'learn_prior': False, 'weibull_shape': 2.0, 'sample_surv': False,
            "activation": None, "survival": True, "epochs":1000}
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
    setup_seed(param["seed"])
    progress = ProgBar(epochs=cfg["epochs"],metrics=["loss","val_output_5_cindex_metric"])
    model = GMM_Survival(**cfg)
    tf.keras.backend.set_value(model.use_t, np.array([1.0]))
    optimizer = tf.keras.optimizers.legacy.Adam(
        learning_rate=param["learning_rate"], decay=0.00001)
    model.compile(optimizer, loss={"output_1": loss}, run_eagerly=True,
                  metrics={"output_5": cindex_metric})
    history = model.fit(gen_tr, validation_data=gen_val,
                        epochs=cfg["epochs"], verbose=0,
                        callbacks=[progress])
    print("loss", history.history['loss'][-1])
    rec, z_sample, p_z_c, p_c_z, risk_scores, lambdas = model.predict(
        (x_val, y_val), batch_size=911, verbose=0)
    risk_scores = np.squeeze(risk_scores)
    ci = cindex(t=y_val[:, 0], e=y_val[:, 1], scores_pred=risk_scores)
    print("ci", ci)
    cis.append(ci)


print(f"Concordance: {np.mean(cis)}(std. {np.std(cis)})")

# # %%
# # # Data generators
# # gen_train = get_gen(x_tr, y_tr, configs, batch_size=256,)
# # gen_val = get_gen(x_val, y_val, configs, batch_size=256, validation=True, shuffle=False)

# # %% [markdown]
# # ### Training & Evaluation

# # %%
# # Fit the model
# # tf.data.experimental.enable_debug_mode()
# # tf.debugging.set_log_device_placement(True)
# # model.fit(gen_tr, validation_data=gen_val, epochs=1000, verbose=1)

# # %%
# # Evaluate the model

# # Don't use survival times during testing
# tf.keras.backend.set_value(model.use_t, np.array([0.0]))

# # Don't use MC samples to predict survival at evaluation
# model.sample_surv = False

# rec, z_sample, p_z_c, p_c_z, risk_scores, lambdas = model.predict((x_tr, y_tr), batch_size=256)
# risk_scores = np.squeeze(risk_scores)
# # Hard cluster assignments
# c_hat = np.argmax(p_c_z, axis=-1)

# # %%
# # Plot cluster-specific Kaplan-Meier curves alongside ground truth
# # plot_bigroup_kaplan_meier(t=t_train, d=d_train, c_=c_hat, c=c_train, legend=True, legend_outside=True)

# # %%
# # Visualise learnt representations
# plot_tsne_by_survival(X=z_sample[:, 0], t=y_tr[:, 0], d=y_tr[:, 1], seed=42, plot_censored=True)

# # %%
# # # Visualise generated samples for each cluster
# # fig, ax = plt.subplots(1, model.num_clusters, figsize=(20,100))

# # grid_size = 5
# # inp_size = (28, 28)

# # for j in range(model.num_clusters):
# #     samples = model.generate_samples(j=j, n_samples=grid_size**2)
# #     cnt = 0
# #     img = None
# #     for k in range(grid_size):
# #         row_k = []
# #         for l in range(grid_size):
# #             row_k.append(np.reshape(samples[0, cnt, :], (inp_size[0], inp_size[1])))
# #             cnt = cnt + 1
# #         if img is None:
# #             img = np.concatenate(row_k, axis=1)
# #         else:
# #             img = np.concatenate([img, np.concatenate(row_k, axis=1)], axis=0)
# #     ax[j].set_title('Cluster ' + str(j + 1))
# #     ax[j].imshow(img, cmap='bone')
# #     ax[j].axis('off')

# # %%
# # Evaluate some metrics on the training data
# # acc = utils.cluster_acc(y_train[:, 2], c_hat)
# # nmi = normalized_mutual_info_score(y_[:, 2], c_hat)
# # ari = adjusted_rand_score(y_train[:, 2], c_hat)

# # %%
# # Now, on the test data
# rec, z_sample, p_z_c, p_c_z, risk_scores, lambdas = model.predict((x_val.values, y_val.values), batch_size=256)
# risk_scores = np.squeeze(risk_scores)
# # Hard cluster assignments
# c_hat = np.argmax(p_c_z, axis=-1)
# # acc = utils.cluster_acc(y_test[:, 2], c_hat)
# # nmi = normalized_mutual_info_score(y_test[:, 2], c_hat)
# # ari = adjusted_rand_score(y_test[:, 2], c_hat)
# ci = cindex(t=y_val.values[:, 0], d=y_val.values[:, 1], scores_pred=risk_scores)
# print('CI (te.): {:.2f}'.format(ci))

# # %%
