from copy import deepcopy
from pprint import pprint
from datasets.radcure import load_radcure, config_radcure
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args
import yaml

from utils.utils import get_purity, get_workdir, setup_env

from main import train
from models.losses import Losses

# Setup
tfk = tf.keras
tfd = tfp.distributions
tfpl = tfp.layers
tfkl = tf.keras.layers


def merge_cfg(base, updates):
    cfg = deepcopy(base)
    for k, v in updates.items():
        if k == 'layers':
            cfg[k] = [int(l) for l in v.split('-')]
        elif k in ['sample_surv', 'learn_prior']:
            cfg[k] = v == "True"
        else:
            cfg[k] = v
    return cfg


def objective(space, seeds, x, y, base_cfg, counter):
    @use_named_args(space)
    def func(**params):
        nonlocal counter
        metrics = []
        for seed in seeds:
            counter[0] += 1
            print(f"run number: {counter[0]}/{counter[1]}")
            cfg = merge_cfg(base_cfg, params)
            cfg["seed"] = seed
            print(cfg)
            model, (x_tr, x_val, y_tr, y_val) = train(x, y, cfg, log=False)
            phe_val = model.get_phenotypes(x_val, y_val)
            metric = get_purity(y_val, phe_val)
            metrics.append(metric)
        print(np.mean(metrics))
        return np.mean(metrics)
    return func


def optimize_k(k, num_calls, seeds, x, y, base_cfg, counter):
    space = [
        Real(1e-4, 5e-3, name='learning_rate'),
        Categorical(['50-100', '100', '100-200', "32-64"], name='layers'),
        Real(1.0, 3.0, name='weibull_shape'),
        Categorical(["True", "False"], name='sample_surv'),
        Integer(2, 16, name='latent_dim'),
        Categorical(["True", "False"], name='learn_prior'),
        Categorical(["selu", "relu"], name="activation")
    ]
    cfg = deepcopy(base_cfg)
    cfg["num_clusters"] = k
    result = gp_minimize(objective(space, seeds, x, y, cfg, counter),
                         space, n_calls=num_calls, verbose=True)
    return result


def convert(value):
    """Converts a NumPy object to a native Python type."""
    if isinstance(value, np.ndarray):
        return value.tolist()  # Convert arrays to lists
    elif isinstance(value, np.generic):
        return value.item()  # Convert NumPy scalars to Python scalars
    else:
        return value


def aspython(value):
    """Recursively convert numpy types to native Python types."""
    if isinstance(value, np.generic):
        return value.item()
    elif isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, list):
        return [aspython(item) for item in value]
    elif isinstance(value, dict):
        return {key: aspython(val) for key, val in value.items()}
    else:
        return value


setup_env()
path = "/u/97/nguyenq10/unix/HUS/cancer-clusters/datasets/radcure/clinical_train.csv"
x, y = load_radcure(path)
# Configuration for the dataset and model
cfg0 = config_radcure()
workdir = get_workdir("radcure/hparams")
cfg0["workdir"] = workdir
K, num_calls, seeds = 8, 100, [3, 4]
nruns = (K-2)*num_calls*len(seeds)
counter = [0, nruns]

all_best_params = []
for k in range(2, K):  # Loop for k=2 to K-1
    result = optimize_k(k, num_calls, seeds, x, y, cfg0, counter)
    top_5 = sorted(zip(result.x_iters, result.func_vals),
                   key=lambda x: x[1])[:5]
    pprint(top_5)
    all_best_params.append(
        {'k': k, 'top_5_params': [{'params': p, 'value': v} for p, v in top_5]})

with open(f'{workdir}/best_params.yaml', 'w') as file:
    yaml.dump(aspython(all_best_params), file, default_flow_style=False)
print(f"Parameters saved to {workdir}best_params.yaml")
