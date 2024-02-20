
from auton_survival.preprocessing import Preprocessor
import pandas as pd

from utils.utils import get_workdir


def load_radcure(path, cat_features=["sex", "disease site", "t stage", "n stage", "hpv combined", "chemo?", "ecog", "stage"], num_features=["age"]):
    df = pd.read_csv(path).drop(columns=["split", "target_binary", "Study ID"])
    df = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)
    df.columns = [col.lower().strip() for col in df.columns]
    outcomes = df[["death", "survival_time"]].rename(
        columns={"death": "event", "survival_time": "time"})
    features = df.drop(columns=["death", "survival_time"])

    preprocessor = Preprocessor(cat_feat_strat='ignore', num_feat_strat='mean')
    x = preprocessor.fit_transform(
        features, cat_feats=cat_features, num_feats=num_features, one_hot=True, fill_value=-1)
    y = outcomes
    print(f'x.shape={x.shape}, y.shape={y.shape}')
    cfg = {"seed": 0, "inp_shape": x.shape[-1], "num_clusters": 2, "latent_dim": 8, 'monte_carlo': 1,
            'learn_prior': False, 'weibull_shape': 2.5, 'sample_surv': False,
            "activation": None, "survival": True, "epochs": 1000, "layers": [50, 100],
            "learning_rate": 1e-3, "activation": "relu", "train_size": 0.7, "workdir": get_workdir("radcure",makedir=False)}
    return x, y, cfg

