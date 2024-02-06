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
from tensorflow.keras.utils import Sequence, to_categorical
import yaml

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

# %%
# from src.radcure import Radcure

from sklearn.model_selection import train_test_split
from auton_survival.preprocessing import Preprocessor


def del_multicol(df):
    return df.drop(columns=df.filter(regex='^N Stage_N[1-9X]').columns).drop(columns=["T Stage_T3/4", "Stage_III/IV", "ECOG_0"]) \
        .drop(columns=df.filter(regex='^Disease').columns).drop(columns="target_binary") \
        .drop(columns=["Sex_Female"])


# df_tr = Radcure.make_data("../data/radcure/clinical_train.csv")
# df_te = Radcure.make_data("/u/97/nguyenq10/unix/HUS/simpy-vadesc/data/radcure/clinical_test.csv")
df = pd.read_csv("./datasets/radcure/clinical_train.csv").drop(
    columns=["split", "target_binary", "Study ID"])
df = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)
df.columns = [col.lower().strip() for col in df.columns]

cat_feats = ["sex", "disease site", "t stage",
             "n stage", "hpv combined", "chemo?", "ecog", "stage"]
num_feats = ["age"]
outcomes = df[["survival_time","death"]].rename(
    columns={"death": "event", "survival_time": "time"})
features = df.drop(columns=["death", "survival_time"])


preprocessor = Preprocessor(cat_feat_strat='ignore', num_feat_strat='mean')
x = preprocessor.fit_transform(features, cat_feats=cat_feats, num_feats=num_feats,
                               one_hot=True, fill_value=-1)
y = outcomes


x_tr, x_val, y_tr, y_val = train_test_split(
    x, y, test_size=0.3, random_state=1)

# Define the times for tuning the model hyperparameters and for evaluating the model
times = np.quantile(y_tr['time'][y_tr['event'] == 1],
                    np.linspace(0.1, 1, 10)).tolist()
print(f'Number of training data points: {len(x_tr)}')
print(f'Number of validation data points: {len(x_val)}')

# %% [markdown]
# ## Vadesc

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
from pprint import pprint
import numpy as np
from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args
from copy import deepcopy
from lifelines.utils import concordance_index
from auton_survival.metrics import survival_regression_metric
from auton_survival.metrics import phenotype_purity


# Define the times for tuning the model hyperparameters and for evaluating the model
times = np.quantile(y_tr['time'][y_tr['event'] == 1],
                    np.linspace(0.1, 1, 10)).tolist()


# Run Bayesian optimization
def bo_hparams(num_clusters, num_calls, x_val, y_val, gen_tr, gen_val):
    # Define the space of hyperparameters to search
    space = [
        # Integer(2,2, name='k'),
        Real(1e-4, 5e-3, name='learning_rate'),
        Categorical(['50-100', '100', '100-100', '32-64', '16-32', '100-200','200-400', '200'],
                name='layers'),
        Real(1.0, 3.0, name='weibull_shape'),
        Categorical(["True", "False"], name="sample_surv"),
        Integer(2, 16, name="latent_dim"),
        # Categorical(["True", "False"], name="learn_prior"),
    ]

    # Define the objective function
    base_cfg = {"inp_shape": 45, "num_clusters": num_clusters, "latent_dim": 8, 'monte_carlo': 1,
                'learn_prior': False, 'weibull_shape': 2.5, 'sample_surv': False,
                "activation": None, "survival": True, "epochs":1000, "layers":[50,100],
                "seed":0,"learning_rate":1e-3}
    # Define reconstruction loss function
    loss = Losses(base_cfg).loss_reconstruction_mse
    

    @use_named_args(space)
    def objective(**params):
        cfg = deepcopy(base_cfg)
        for k, v in params.items():
            if k == 'layers':
                cfg[k] = [int(layer)
                        for layer in v.split('-')]  # Convert string to list
            elif k == "sample_surv" or k == "learn_prior":
                cfg[k] = bool(v)
            elif k in cfg:
                cfg[k] = v

        setup_seed(cfg["seed"])
        progress = ProgBar(epochs=cfg["epochs"],metrics=["loss","val_output_5_cindex_metric"])
        model = GMM_Survival(**cfg)
        tf.keras.backend.set_value(model.use_t, np.array([1.0]))
        optimizer = tf.keras.optimizers.legacy.Adam(
            learning_rate=cfg["learning_rate"], decay=0.00001)
        model.compile(optimizer, loss={"output_1": loss}, run_eagerly=False,
                      metrics={"output_5": cindex_metric})
        print(cfg)
        history = model.fit(gen_tr, validation_data=gen_val,
                            epochs=cfg["epochs"], verbose=0,
                            callbacks=[progress])
        _, _, _, p_c_z, risks, _ = model.predict((x_val.values, y_val.values))
        risks = np.squeeze(risks)
        phenotypes = np.argmax(p_c_z, -1)
        # Estimate the Integrated Brier Score at event horizons of 1, 2 and 5 years
        metric_val1 = phenotype_purity(phenotypes_train=phenotypes, outcomes_train=y_val,
                                       phenotypes_test=None, outcomes_test=None,
                                       strategy='integrated', horizons=[6],
                                       bootstrap=None)

        # Return negative metric for minimization
        print(np.mean(metric_val1))
        return np.mean(metric_val1)

    result = gp_minimize(objective, space, n_calls=num_calls, random_state=0,)

# Extract the best parameters
    best_params = {dim.name: result.x[i] for i, dim in enumerate(space)}

# Pair each set of parameters with its corresponding function value
    params_and_func_vals = [(result.x_iters[i], result.func_vals[i])
                        for i in range(len(result.x_iters))]

# Sort the pairs based on the function value
    sorted_params_and_vals = sorted(params_and_func_vals, key=lambda x: x[1])

# Get the top 5 best parameter sets
    top_5_params = sorted_params_and_vals[:5]

    print("=================Top 5 Best Parameters=================")
    for params, val in top_5_params:
        pprint({"params": params, "objective_value": val})

    print("Best Parameters:", best_params)
    return top_5_params

K = 3  # For example, to include k=2 and k=3
num_calls = 10
all_best_params = []
for k in range(2, K):  # Now this will include k=2 and k=3
    best_params_k = bo_hparams(k, num_calls, x_val, y_val, gen_tr, gen_val)
    all_best_params.append((k, best_params_k))
    
for k, top_5_params in all_best_params:
    print(f"=================k = {k}=================")
    for params, val in top_5_params:
        pprint({"params": params, "objective_value": val})
        
# Now, to save this to a YAML file
filename = 'best_params.yaml'
with open(filename, 'w') as file:
    yaml.dump(all_best_params, file, default_flow_style=False)
    
print(f"Parameters saved to {filename}")
