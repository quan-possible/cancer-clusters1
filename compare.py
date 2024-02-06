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
tfd = tfp.distributions
tfkl = tf.keras.layers
tfpl = tfp.layers
tfk = tf.keras
# VaDeSC model

# Set Seaborn style for a cleaner look
sns.set(style="whitegrid")

os.chdir("/u/97/nguyenq10/unix/HUS/vadesc-torch")
print(os.getcwd())


# VaDeSC model


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
features = df.drop(columns=["death","survival_time"])

from auton_survival.preprocessing import Preprocessor

preprocessor = Preprocessor(cat_feat_strat='ignore', num_feat_strat= 'mean') 
x = preprocessor.fit_transform(features, cat_feats=cat_feats, num_feats=num_feats,
                                one_hot=True, fill_value=-1)
y = outcomes


from sklearn.model_selection import train_test_split


x_tr, x_val, y_tr, y_val = train_test_split(x, y, test_size=0.3, random_state=1) 

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

(next(iter(gen_tr))[0][0].shape,next(iter(gen_tr))[0][1].shape, )

# %%
"Model"
from copy import deepcopy
from utils.utils import get_workdir

import torch
from auton_survival.metrics import phenotype_purity, survival_regression_metric
from lifelines.utils import concordance_index
from sklearn.model_selection import ParameterGrid
from torch.utils.tensorboard.writer import SummaryWriter

param_grid = {'num_clusters': [2],
              'learning_rate': [1e-3],
              "seed": [0]
              }

params = ParameterGrid(param_grid)

# Define the times for tuning the model hyperparameters and for evaluating the model
# times = np.quantile(y_tr['time'][y_tr['event']==1], np.linspace(0.1, 1, 10)).tolist()
# Perform hyperparameter tuning
models = []
base_cfg = {"inp_shape": 45, "num_clusters": 2, "latent_dim": 8, 'monte_carlo': 1,
            'learn_prior': False, 'weibull_shape': 2.5, 'sample_surv': False,
            "activation": None, "survival": True, "epochs":1000, "layers":[50,100]}
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
    tb_cb = tensorboard_callback = TensorBoard(log_dir=get_workdir())
    tf.keras.backend.set_value(model.use_t, np.array([1.0]))
    optimizer = tf.keras.optimizers.legacy.Adam(
        learning_rate=param["learning_rate"], decay=0.00001)
    model.compile(optimizer, loss={"output_1": loss}, run_eagerly=True,
                  metrics={"output_5": cindex_metric})
    history = model.fit(gen_tr, validation_data=gen_val,
                        epochs=cfg["epochs"], verbose=0,
                        callbacks=[progress,])
    print("loss", history.history['loss'][-1])
    rec, z_sample, p_z_c, p_c_z, risks, lambdas = model.predict((x_val.values, y_val.values))
    risks = np.squeeze(risks)
    phenotypes = np.argmax(p_c_z, -1)
    metric_val1 = phenotype_purity(phenotypes_train=phenotypes, outcomes_train=y_val, 
                                    phenotypes_test=None, outcomes_test=None,
                                    strategy='integrated', horizons=[6], 
                                    bootstrap=None)
    # metric_val1 = survival_regression_metric(
    #     'ibs', y_val, predictions_val, times, y_tr)
    # metric_val2 = concordance_index(y_val.time, risk, y_val.event)
    print(f"purity={metric_val1}")

    models.append([metric_val1, model, param])
    ci = cindex(t=y_val.values[:, 0], e=y_val.values[:, 1], scores_pred=risks)
    print("ci", ci)
    cis.append(ci)


print(f"Concordance: {np.mean(cis)}(std. {np.std(cis)})")

# Select the best model based on the mean metric value computed for the validation set
models = sorted(models, key=lambda x: x[0], reverse=True)
best_model = models[0][1]
print(models)

# %%
from auton_survival import reporting
from auton_survival.metrics import phenotype_purity

# Estimate the probability of event-free survival for phenotypes using the Kaplan Meier estimator.
rec, z_sample, p_z_c, p_c_z, risk_scores, lambdas = best_model.predict((x_val.values, y_val.values), verbose=0)
# phenotypes = np.argmax(p_c_z, -1)
phenotypes = np.random.randint(0,1,len(x_val))
reporting.plot_kaplanmeier(y_val, groups=phenotypes)

plt.xlabel('Time (Years)')
plt.ylabel('Event-Free Survival Probability')
plt.legend(['Phenotype 1', 'Phenotype 2'], loc="upper right")
plt.show()

# %%
plt.hist(phenotypes)
# Display the plot
plt.show()


plot_latents_(model.torch_model,x=torch.from_numpy(x_tr.values).double(), t=torch.from_numpy(y_tr.time.values).double())

None

# %%
from auton_survival.models.dcm import DeepCoxMixtures
from auton_survival.metrics import survival_regression_metric
from sklearn.model_selection import ParameterGrid

# Define parameters for tuning the model
param_grid1 = {'k' : [2],
              'learning_rate' : [1e-3, 1e-4],
              'layers' : [[50, 100],[100]]
             }

params1 = ParameterGrid(param_grid1)

# Define the times for tuning the model hyperparameters and for evaluating the model
times = np.quantile(y_tr['time'][y_tr['event']==1], np.linspace(0.1, 1, 10)).tolist()

# Perform hyperparameter tuning 
models1 = []
for param in params1:
    model = DeepCoxMixtures(k=param['k'], layers=param['layers'],)
    
    # The fit method is called to train the model
    model.fit(x_tr, y_tr.time, y_tr.event, iters = 100, learning_rate=param['learning_rate'])

    phenotypes = np.argmax(model.predict_latent_z(x_val),-1)
    # Estimate the integrated Brier Score at event horizons of 1, 2 and 5 years
    metric_val1 = phenotype_purity(phenotypes_train=phenotypes, outcomes_train=y_val, 
                                    phenotypes_test=None, outcomes_test=None,
                                    strategy='integrated', horizons=[6], 
                                    bootstrap=None)
    # Obtain survival probabilities for validation set and compute the integrated Brier Score 
    # predictions_val = model.predict_survival(x_val, times)
    # metric_val2 = survival_regression_metric('ctd', y_val, predictions_val, times, y_tr)
    # metric_val1 = survival_regression_metric('ibs', y_val, predictions_val, times, y_tr)
    models1.append([metric_val1, model, param])
    
models1 = sorted(models1, key=lambda x: x[0], reverse=False)
model = models1[0][1]
print(models1)

# %%
print(models1)

# %%
from auton_survival import reporting
import matplotlib.pyplot as plt
from auton_survival.models.dcm.dcm_utilities import predict_latent_z

latent_z_prob = model.predict_latent_z(x_val)
phenotypes = np.argmax(latent_z_prob, axis=-1)
# Estimate the probability of event-free survival for phenotypes using the Kaplan Meier estimator.
reporting.plot_kaplanmeier(y_val, np.array(phenotypes))

plt.xlabel('Time (Years)')
plt.ylabel('Event-Free Survival Probability')
plt.legend(['Phenotype 1', 'Phenotype 2'], loc="upper right")
plt.show()

# Estimate the integrated Brier Score at event horizons of 1, 2 and 5 years
metric = phenotype_purity(phenotypes_train=phenotypes, outcomes_train=y_val, 
                                phenotypes_test=None, outcomes_test=None,
                                strategy='integrated', horizons=[6], 
                                bootstrap=None)
metric

# %%
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=1)
x_tsne = tsne.fit_transform(x_val)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=x_tsne[:, 0], y=x_tsne[:, 1], hue=phenotypes, size=y_val.time)
plt.xlabel('TSNE Component 1')
plt.ylabel('TSNE Component 2')
plt.title('TSNE Plot with Phenotypes')
plt.legend(title='Phenotypes', loc='best')
plt.show()



