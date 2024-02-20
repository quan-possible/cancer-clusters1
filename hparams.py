from copy import deepcopy
from pprint import pprint
import sys
from datasets.radcure import load_radcure
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


def objective(space, seeds, x, y, base_cfg):
    @use_named_args(space)
    def func(**params):
        metrics = []
        for seed in seeds:
            cfg = merge_cfg(base_cfg, params)
            cfg["seed"] = seed
            print(cfg)
            x_tr, x_val, y_tr, y_val = train_test_split(
                x, y, train_size=cfg["train_size"], random_state=cfg["seed"])
            model, _ = train(
                x_tr, y_tr, cfg, validation_data=(x_val,y_val), log=False)
            phe_val = model.get_phenotypes(x_val, y_val)
            metric = get_purity(y_val, phe_val)
            metrics.append(metric)
        print(np.mean(metrics))
        return np.mean(metrics)
    return func


def optimize_k(num_calls, seeds, x, y, base_cfg):
    space = [
        Categorical(['50-100', '100', '100-200'], name='layers'),
        Real(1.0, 3.0, name='weibull_shape'),
        Integer(2, 16, name='latent_dim'),
        Categorical(["selu", "relu"], name="activation")
    ]
    cfg = deepcopy(base_cfg)
    result = gp_minimize(objective(space, seeds, x, y, cfg),
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
path = "/u/97/nguyenq10/unix/HUS/cancer-clusters/datasets/radcure/clinical1.csv"
x, y, cfg0 = load_radcure(path)
# Configuration for the dataset and model

if len(sys.argv) > 1:
    cfg0["num_clusters"] = int(sys.argv[1])
workdir0 = "radcure/hparams"
workdir = get_workdir(workdir0, f"k{cfg0['num_clusters']}")
cfg0["workdir"] = workdir
num_calls, seeds = 100, [0]

best_params = []
result = optimize_k(num_calls, seeds, x, y, cfg0)
top_5 = sorted(zip(result.x_iters, result.func_vals),
               key=lambda x: x[1])[:5]
pprint(top_5)
best_params.append([{'params': p, 'value': v} for p, v in top_5])

with open(f'{workdir}/best_params.yaml', 'w') as file:
    yaml.dump(aspython(best_params), file, default_flow_style=False)
print(f"Parameters saved to {workdir}/best_params.yaml")
