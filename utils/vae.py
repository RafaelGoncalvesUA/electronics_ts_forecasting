import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import tensorflow as tf
import keras
from keras import ops
from keras import layers

from keras.callbacks import EarlyStopping
import pandas as pd
from tqdm.keras import TqdmCallback


class Sampling(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seed_generator = keras.random.SeedGenerator(1337)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = ops.shape(z_mean)[0]
        dim = ops.shape(z_mean)[1]
        epsilon = keras.random.normal(shape=(batch, dim), seed=self.seed_generator)
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon


class CustomVAE(keras.Model):
    def __init__(self, base_dir='my_models/models', act='swish', init='glorot_uniform', **kwargs):
        super().__init__(**kwargs)
        self.act = act
        self.init = init
        self.base_dir = base_dir
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")


    def build_encoder(self, dims):
        n_stacks = len(dims) - 1

        encoder_inputs = keras.Input(shape=(dims[0],))
        x = encoder_inputs
        
        for i in range(n_stacks-1):
            x = layers.Dense(dims[i + 1], activation=self.act, kernel_initializer=self.init, name='encoder_%d' % i)(x)

        z_mean = layers.Dense(dims[-1], name="z_mean")(x)
        z_log_var = layers.Dense(dims[-1], name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])

        self.encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")


    def build_decoder(self, dims):
        latent_inputs = keras.Input(shape=(dims[-1],))
        x = latent_inputs

        for i in range(len(dims)-1, 0, -1):
            x = layers.Dense(dims[i], activation=self.act, kernel_initializer=self.init, name='decoder_%d' % i)(x)

        decoder_outputs = layers.Dense(dims[0], activation='linear', name='output')(x)
        self.decoder = keras.Model(latent_inputs, decoder_outputs, name='decoder')


    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]
    

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = ops.mean(ops.sum(ops.square(data - reconstruction), axis=1))
            kl_loss = -0.5 * (1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))
            kl_loss = ops.mean(ops.sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


    def build_autoencoder(self, dims, act=None, init=None):
        """
        Fully connected auto-encoder model, symmetric.
        Arguments:
            dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
                The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
            act: activation, not applied to Input, Hidden and Output layers
        return:
            (ae_model, encoder_model), Model of autoencoder and model of encoder
        """
        if act is None:
            act = self.act

        if init is None:
            init = self.init

        self.build_encoder(dims)
        self.build_decoder(dims)


    def train(self, df, to_forecast, lookback, out_steps, latent_dim=5, train_again=False):
        columns_to_keep = [f"{to_forecast}_future{i}" for i in range(1, out_steps+1)] + ['Hour', 'Day Type']
        self.to_forecast = to_forecast
        self.out_steps = out_steps
        self.latent_dim = latent_dim
        
        df_keep = df[columns_to_keep]
        df_reduce = df.drop(columns_to_keep, axis=1)

        encoder_path = f'{self.base_dir}/encoder_{to_forecast.replace("/", "")}_l{lookback}_f{out_steps}.keras'

        if train_again:
            print('-> Training encoder...')
            xs = df_reduce
            dims = [xs.shape[-1], 64, 32, latent_dim]
            
            self.build_autoencoder(dims)

            self.compile(optimizer=keras.optimizers.Adam())
            self.fit(xs, batch_size=64, epochs=2000, verbose=0, callbacks=[TqdmCallback(verbose=0),
                    EarlyStopping(monitor='loss', patience=10, verbose=0)])

            self.encoder.save(encoder_path)
        else:
            print('-> Loading encoder from: {}'.format(encoder_path))
            self.encoder = tf.keras.models.load_model(encoder_path, safe_mode=False)
            xs = df_reduce

        reduced_df = pd.DataFrame(self.encoder.predict(xs)[-1], columns=['Z{}'.format(i) for i in range(latent_dim)])
        return pd.concat([df_keep, reduced_df], axis=1)
    
    def predict(self, df):
        columns_to_keep = [f"{self.to_forecast}_future{i}" for i in range(1, self.out_steps+1)] + ['Hour', 'Day Type']
        
        df_keep = df[columns_to_keep]
        df_reduce = df.drop(columns_to_keep, axis=1)
        
        reduced_df = pd.DataFrame(self.encoder.predict(df_reduce)[-1], columns=['Z{}'.format(i) for i in range(self.latent_dim)])
        return pd.concat([df_keep, reduced_df], axis=1)
