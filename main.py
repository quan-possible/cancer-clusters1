from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.callbacks import TensorBoard

from models.losses import Losses
from models.model import GMM_Survival
from utils.eval_utils import cindex_metric, purity_metric
from utils.utils import (ProgBar, get_workdir, setup_env,
                         setup_seed)

from datasets.radcure import  load_radcure

# TensorFlow Probability shortcuts
tfd = tfp.distributions
tfkl = tf.keras.layers
tfpl = tfp.layers
tfk = tf.keras


def train(x, y, config, validation_data=None, log=True, return_history=True):
    # hparams
    seed, epochs, train_size, lr, workdir = config["seed"], config[
        "epochs"], config["train_size"], config["learning_rate"], config["workdir"]
    setup_seed(seed)
    # data
    if validation_data is None:
        x_tr, x_val, y_tr, y_val = train_test_split(
            x, y, train_size=train_size, random_state=seed)
    else:
        x_tr, y_tr, x_val, y_val = x, y, *validation_data
    def get_input(x, y): return ((x, y), {"output_1": x, "output_5": y})
    # model
    model = GMM_Survival(**config)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = Losses(config).loss_reconstruction_mse
    model.compile(optimizer, loss={"output_1": loss}, metrics={
        "output_5": cindex_metric, "output_7": purity_metric})
    # callbacks
    progress = ProgBar(epochs=epochs, metrics=[
                       "loss", "val_output_5_cindex_metric", "val_output_7_purity_metric"])
    tb = TensorBoard(log_dir=workdir) if log else None
    cbs = [progress, tb] if log else [progress]
    # train
    history = model.fit(*get_input(x_tr, y_tr), validation_data=get_input(
        x_val, y_val), epochs=epochs, callbacks=cbs, batch_size=len(x_tr), verbose=0)
    return model, history if return_history else model


if __name__ == "__main__":
    setup_env()
    filepath = "./datasets/radcure/clinical_train.csv"

    x, y, cfg0 = load_radcure(filepath)
    model, history = train(x, y, cfg0)
