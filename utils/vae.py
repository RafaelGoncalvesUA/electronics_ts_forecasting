import tensorflow as tf
import keras
from keras import backend as K
from keras.layers import Dense, Input, Lambda
from keras.models import Model
from keras.callbacks import EarlyStopping
import pandas as pd
from tqdm.keras import TqdmCallback
import keras.backend as K

tf.keras.utils.set_random_seed(42)

def sampling(args):
    z_mean, z_log_sigma = args
    latent_dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=1.)
    return z_mean + K.exp(z_log_sigma) * epsilon

class CustomVAE():
    def __init__(self, base_dir='my_models/models', act='swish', init='glorot_uniform'):
        self.act = act
        self.init = init
        self.base_dir = base_dir

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

        n_stacks = len(dims) - 1
        # input
        inputs = Input(shape=(dims[0],), name='input')
        x = inputs
        # internal layers in encoder
        for i in range(n_stacks-1):
            x = Dense(dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(x)

        # hidden layer
        #encoded = Dense(dims[-1], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))(x)  # hidden layer, features are extracted from here
        #--- Custom Latent Space Layer
        z_mean = Dense(units=dims[-1], name='Z-Mean')(x) # Mean component
        z_log_sigma = Dense(units=dims[-1], name='Z-Log-Sigma')(x) # Standard deviation component
        z = Lambda(sampling, name='Z-Sampling-Layer')([z_mean, z_log_sigma]) # Z sampling layer

        latent_inputs = Input(shape=(dims[-1],), name='Input-Z-Sampling')
        x = latent_inputs
        # internal layers in decoder
        for i in range(n_stacks-1, 0, -1):
            x = Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(x)

        #--- Output Layer Decoder
        outputs = Dense(dims[0], activation='linear', name='output')(x)

        # Instantiate a VAE model
        encoder = Model(inputs, [z_mean, z_log_sigma, z], name='Encoder-Model')
        decoder = Model(latent_inputs, outputs, name='Decoder-Model')

        # Define outputs from a VAE model by specifying how the encoder-decoder models are linked
        output_vae = decoder(encoder(inputs)[2]) # note, outputs available from encoder model are z_mean, z_log_sigma and z. We take z by specifying [2]

        vae = Model(inputs=inputs, outputs=output_vae, name='VAE')

        # Reconstruction loss compares inputs and outputs and tries to minimise the difference
        r_loss = dims[0] * keras.losses.mse(inputs, output_vae)  # use MSE

        # KL divergence loss compares the encoded latent distribution Z with standard Normal distribution and penalizes if it's too different
        kl_loss =  -0.5 * K.sum(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis = 1)

        # The VAE loss is a combination of reconstruction loss and KL loss
        vae_loss = K.mean(r_loss + kl_loss)

        # Add loss to the model and compile it
        vae.add_loss(vae_loss)
        vae.compile(optimizer='adam')

        return vae, encoder
    
    def fit(self, df, to_forecast, lookback, out_steps, latent_dim=5, train_again=False):
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
            autoencoder, self.encoder = self.build_autoencoder(dims)
            autoencoder.fit(xs, xs, batch_size=64, epochs=2000, verbose=0, callbacks=[TqdmCallback(verbose=0),
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
