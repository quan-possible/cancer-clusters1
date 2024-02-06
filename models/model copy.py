"""
VaDeSC model.
"""
import tensorflow as tf
import tensorflow_probability as tfp
import os

from models.networks import (
    VGGEncoder, VGGDecoder, Encoder_small, Decoder_small)
import tensorflow as tf
from tensorflow.keras import layers, models


from utils.utils import weibull_scale, weibull_log_pdf, tensor_slice

# Pretrain autoencoder
checkpoint_path = "autoencoder/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

tfd = tfp.distributions
tfkl = tf.keras.layers
tfpl = tfp.layers
tfk = tf.keras


class GMM_Survival(tf.keras.Model):
    def __init__(self, **kwargs):
        super(GMM_Survival, self).__init__(name="GMM_Survival")
        self.encoded_size = kwargs['latent_dim']
        self.num_clusters = kwargs['num_clusters']
        self.inp_shape = kwargs['inp_shape']
        self.activation = kwargs['activation']
        self.survival = kwargs['survival']
        self.s = kwargs['monte_carlo']
        self.sample_surv = kwargs['sample_surv']
        self.learn_prior = kwargs['learn_prior']
        self.layers_ = kwargs['layers']
        # if isinstance(self.inp_shape, list):
        #     self.encoder = VGGEncoder(encoded_size=self.encoded_size)
        #     self.decoder = VGGDecoder(
        #         input_shape=[256, 256, 1], activation='none')
        # elif self.inp_shape <= 100:
        #     self.encoder = Encoder_small(self.encoded_size)
        #     self.decoder = Decoder_small(self.inp_shape, self.activation)
        # else:
        #     self.encoder = Encoder(self.encoded_size)
        #     self.decoder = Decoder(self.inp_shape, self.activation)
        self.encoder, self.decoder = self.init_nn(
            self.inp_shape, self.layers_, self.encoded_size, activation='relu', bias=True)
        self.c_mu = tf.Variable(tf.initializers.GlorotNormal()(
            shape=[self.num_clusters, self.encoded_size]), name='mu')
        self.log_c_sigma = tf.Variable(tf.initializers.GlorotNormal()(
            [self.num_clusters, self.encoded_size]), name="sigma")
        # Cluster-specific survival model parameters
        self.c_beta = tf.Variable(tf.initializers.GlorotNormal()(shape=[self.num_clusters, self.encoded_size + 1]),
                                  name='beta')
        # Weibull distribution shape parameter
        self.weibull_shape = kwargs['weibull_shape']
        if self.learn_prior:
            self.prior_logits = tf.Variable(
                tf.ones([self.num_clusters]), name="prior")
        else:
            self.prior = tf.constant(
                tf.ones([self.num_clusters]) * (1 / self.num_clusters))
        self.use_t = tf.Variable([1.0], trainable=False)

    def call(self, inputs, training=True):
        if training:
            use_t_value = 1.0
        else:
            use_t_value = 0.0
        # self.use_t = use_t_value * self.use_t
        # NB: inputs have to include predictors/covariates/features (x), time-to-event (t), and
        # event indicators (e). e[i] == 1 if the i-th event is a death, and e[i] == 0 otherwise.
        x, y = inputs
        t = y[:, 0]
        e = y[:, 1]
        log_z_sigma, z_sample, p_z_c, prior, lambda_z_c, p_t_z_c, p_c_z = self.encode(
            x, t, e, training)

        self.get_loss(log_z_sigma, p_z_c, prior, p_t_z_c, p_c_z)

        dec = self.decoder(z_sample)

        # Evaluate risk scores based on hard clustering assignments
        # Survival time may ba unobserved, so a special procedure is needed when time is not observed...
        p_z_c = p_z_c[0]    # take the first sample
        p_c_z = p_c_z[0]

        lambda_z_c, p_c_z, risks = self.get_risks(
            p_z_c, prior, lambda_z_c, p_c_z, training)

        p_z_c = tf.cast(p_z_c, tf.float64)

        if isinstance(self.inp_shape, list):
            dec = tf.transpose(dec, [1, 0, 2, 3, 4])
        else:
            dec = tf.transpose(dec, [1, 0, 2])
        z_sample = tf.transpose(z_sample, [1, 0, 2])
        risks = tf.expand_dims(risks, -1)
        return dec, z_sample, p_z_c, p_c_z, risks, lambda_z_c

    def get_risks(self, p_z_c, prior, lambda_z_c, p_c_z, training):
        if self.survival:
            lambda_z_c = lambda_z_c[0]  # Take the first sample
            # Use Bayes rule to compute p(c|z) instead of p(c|z,t), since t is unknown
            p_c_z_nt = tf.math.log(
                tf.cast(prior, tf.float64) + 1e-60) + tf.cast(p_z_c, tf.float64)
            p_c_z_nt = tf.nn.log_softmax(p_c_z_nt, axis=-1)
            p_c_z_nt = tf.math.exp(p_c_z_nt)
            inds_nt = tf.dtypes.cast(tf.argmax(p_c_z_nt, axis=-1), tf.int32)
            risks_nt = tensor_slice(target_tensor=tf.cast(
                lambda_z_c, tf.float64), index_tensor=inds_nt)

            inds = tf.dtypes.cast(tf.argmax(p_c_z, axis=-1), tf.int32)
            risks_t = tensor_slice(
                target_tensor=lambda_z_c, index_tensor=inds)

            # p_c_z = tf.cond(use_t_value < 0.5, lambda: p_c_z_nt, lambda: p_c_z)
            p_c_z = p_c_z if training else p_c_z_nt
            risks = risks_t if training else risks_nt
            # risks = tf.cond(
            #     use_t_value < 0.5, lambda: risks_nt, lambda: risks_t)
        else:
            inds = tf.dtypes.cast(tf.argmax(p_c_z, axis=-1), tf.int32)
            risks = tensor_slice(target_tensor=p_c_z, index_tensor=inds)
            lambda_z_c = risks
        return lambda_z_c, p_c_z, risks

    def encode(self,  x, t=None, e=None, training=True):
        enc_input = x
        z_mu, log_z_sigma = self.encoder(enc_input)
        tf.debugging.check_numerics(z_mu, message="z_mu")
        z = tfd.MultivariateNormalDiag(
            loc=z_mu, scale_diag=tf.math.sqrt(tf.math.exp(log_z_sigma)))
        if training:
            z_sample = z.sample(self.s)
        else:
            z_sample = tf.expand_dims(z_mu, 0)
        tf.debugging.check_numerics(self.c_mu, message="c_mu")
        tf.debugging.check_numerics(self.log_c_sigma, message="c_sigma")
        c_sigma = tf.math.exp(self.log_c_sigma)

        # p(z|c)
        p_z_c = tf.stack([tf.math.log(
            tfd.MultivariateNormalDiag(loc=tf.cast(self.c_mu[i, :], tf.float64),
                                       scale_diag=tf.math.sqrt(tf.cast(c_sigma[i, :], tf.float64))).prob(
                tf.cast(z_sample, tf.float64)) + 1e-60) for i in range(self.num_clusters)], axis=-1)
        tf.debugging.check_numerics(p_z_c, message="p_z_c")

        # prior p(c)
        if self.learn_prior:
            prior_logits = tf.math.abs(self.prior_logits)
            norm = tf.math.reduce_sum(prior_logits, keepdims=True)
            prior = prior_logits / (norm + 1e-60)
        else:
            prior = self.prior
        tf.debugging.check_numerics(prior, message="prior")

        if self.survival:
            # Compute Weibull distribution's scale parameter, given z and c
            tf.debugging.check_numerics(self.c_beta, message="c_beta")
            if self.sample_surv:
                lambda_z_c = tf.stack([weibull_scale(x=z_sample, beta=self.c_beta[i, :])
                                       for i in range(self.num_clusters)], axis=-1)
            else:
                lambda_z_c = tf.stack([weibull_scale(x=tf.stack([z_mu for i in range(self.s)], axis=0),
                                                     beta=self.c_beta[i, :]) for i in range(self.num_clusters)], axis=-1)
            tf.debugging.check_numerics(lambda_z_c, message="lambda_z_c")

            # Evaluate p(t|z,c), assuming t|z,c ~ Weibull(lambda_z_c, self.weibull_shape)
            p_t_z_c = tf.stack([weibull_log_pdf(t=t, e=e, lmbd=lambda_z_c[:, :, i], k=self.weibull_shape)
                                for i in range(self.num_clusters)], axis=-1) if t is not None and e is not None else 0
            p_t_z_c = tf.clip_by_value(p_t_z_c, -1e+64, 1e+64)
            tf.debugging.check_numerics(p_t_z_c, message="p_t_z_c")
            p_c_z = tf.math.log(tf.cast(prior, tf.float64) +
                                1e-60) + tf.cast(p_z_c, tf.float64) + p_t_z_c
        else:
            p_c_z = tf.math.log(tf.cast(prior, tf.float64) +
                                1e-60) + tf.cast(p_z_c, tf.float64)
            lambda_z_c, p_t_z_c = None, None

        p_c_z = tf.nn.log_softmax(p_c_z, axis=-1)
        p_c_z = tf.math.exp(p_c_z)
        tf.debugging.check_numerics(p_c_z, message="p_c_z")
        return log_z_sigma, z_sample, p_z_c, prior, lambda_z_c, p_t_z_c, p_c_z

    def get_loss(self, log_z_sigma, p_z_c, prior, p_t_z_c, p_c_z):
        if self.survival:
            loss_survival = - \
                tf.reduce_sum(tf.multiply(
                    p_t_z_c, tf.cast(p_c_z, tf.float64)), axis=-1)
            tf.debugging.check_numerics(loss_survival, message="loss_survival")

        loss_clustering = - tf.reduce_sum(tf.multiply(tf.cast(p_c_z, tf.float64), tf.cast(p_z_c, tf.float64)),
                                          axis=-1)

        # loss_prior = - tf.math.reduce_sum(tf.math.xlogy(tf.cast(p_c_z, tf.float64), 1e-60 +
        #                                                 tf.cast(prior, tf.float64)), axis=-1)

        # TODO: Remember to change back
        loss_prior = - 10 * (tf.math.reduce_sum(tf.math.xlogy(tf.cast(p_c_z, tf.float64), 1e-60 +
                                                              tf.cast(prior, tf.float64)), axis=-1))

        loss_variational_1 = - 1 / 2 * tf.reduce_sum(log_z_sigma + 1, axis=-1)

        loss_variational_2 = tf.math.reduce_sum(tf.math.xlogy(tf.cast(p_c_z, tf.float64),
                                                              1e-60 + tf.cast(p_c_z, tf.float64)), axis=-1)

        tf.debugging.check_numerics(loss_clustering, message="loss_clustering")
        tf.debugging.check_numerics(loss_prior, message="loss_prior")
        tf.debugging.check_numerics(
            loss_variational_1, message="loss_variational_1")
        tf.debugging.check_numerics(
            loss_variational_2, message="loss_variational_2")

        if self.survival:
            self.add_loss(tf.math.reduce_mean(loss_survival))
        self.add_loss(tf.math.reduce_mean(loss_clustering))
        self.add_loss(tf.math.reduce_mean(loss_prior))
        self.add_loss(tf.math.reduce_mean(loss_variational_1))
        self.add_loss(tf.math.reduce_mean(loss_variational_2))

        # Logging metrics in TensorBoard
        self.add_metric(loss_clustering, name='loss_clustering',
                        aggregation="mean")
        self.add_metric(loss_prior, name='loss_prior', aggregation="mean")
        self.add_metric(loss_variational_1,
                        name='loss_variational_1', aggregation="mean")
        self.add_metric(loss_variational_2,
                        name='loss_variational_2', aggregation="mean")
        if self.survival:
            self.add_metric(loss_survival, name='loss_survival',
                            aggregation="mean")

    def generate_samples(self, j, n_samples):
        z = tfd.MultivariateNormalDiag(loc=self.c_mu[j, :], scale_diag=tf.math.sqrt(
            tf.math.exp(self.log_c_sigma[j, :])))
        z_sample = z.sample(n_samples)
        dec = self.decoder(tf.expand_dims(z_sample, 0))
        return dec

    def init_nn(self, input_dim, layers_, latent_dim, activation='relu', bias=True):
        # Define activation function
        # if activation == 'relu6':
        #     act = tfkl.ReLU(max_value=6)
        # elif activation == 'relu':
        #     act = tfkl.ReLU()
        # elif activation == 'selu':
        #     act = tfkl.SELU()
        # elif activation == 'tanh':
        #     act = tfkl.Activation('tanh')

        encoder = Encoder(
            layers_=layers_, latent_dim=latent_dim, activation=activation)
        decoder = Decoder(layers_=layers_, input_dim=input_dim, activation=activation)

        return encoder, decoder


class Encoder(models.Model):
    def __init__(self, layers_, latent_dim, activation='relu', bias=True):
        super(Encoder, self).__init__()
        self.layers_ = tfk.Sequential([tfkl.Dense(layer, activation=activation, use_bias=bias)
                        for layer in layers_])
        self.mu = tfkl.Dense(latent_dim, use_bias=bias)
        self.sigma = tfkl.Dense(latent_dim, use_bias=bias)

    def call(self, inputs):
        x = self.layers_(inputs)
        return self.mu(x), self.sigma(x)


class Decoder(models.Model):
    def __init__(self, layers_, input_dim, activation='relu', bias=True):
        super(Decoder, self).__init__()
        # Reverse the layer sizes for the decoder
        self.layers_ = tfk.Sequential([tfkl.Dense(layer, activation=activation, use_bias=bias)
                        for layer in reversed(layers_)])
        # Reconstruct the original input dimension
        self.out = tfkl.Dense(input_dim, use_bias=bias)

    def call(self, inputs):
        x = self.layers_(inputs)
        return self.out(x)


if __name__ == "__main__":
    input_dim = 16
    layers_ = [50,100]
    latent_dim = 16
    activation = "relu"
    encoder = Encoder(layers_=layers_, latent_dim=latent_dim, activation=activation)
    decoder = Decoder(input_dim=input_dim, layers_=layers_, latent_dim=latent_dim, activation=activation)

    # Dummy input for encoder and decoder to build them
    dummy_input_enc = tf.zeros((1, input_dim))
    _ = encoder(dummy_input_enc)

    dummy_input_dec = tf.zeros((1, latent_dim))
    _ = decoder(dummy_input_dec)

    def print_model_details(model, indent=0):
        indent_str = ' ' * indent
        if hasattr(model, 'layers'):
            print(indent_str + model.name)
            for layer in model.layers:
                # Check if this layer is a model itself and print its summary
                if hasattr(layer, 'layers'):
                    print_model_details(layer, indent=indent+2)
                else:
                    print(indent_str + ' -', layer.name, layer.output_shape)
        else:
            print(indent_str + ' -', model.name, model.output_shape)

    print_model_details(best_model)
