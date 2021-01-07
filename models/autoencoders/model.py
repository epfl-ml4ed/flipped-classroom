#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf

from models.autoencoders.layers import dense_unit, recurrent_unit, convolutional_unit
from models.autoencoders.layers import SampleNormal

def vae_conv2lstm(config):
    # Create a simple variational autoencoder with a convolutional encoder and a lstm decoder inspired by PixelCNN https://arxiv.org/abs/1606.05328
    n_features = config['n_features']
    n_timesteps = int(n_features / config['obs_per_timestep'])
    shape = [None, n_timesteps, config['obs_per_timestep']]

    # Encoder
    x = tf.keras.layers.Input(shape=(shape[1], shape[2]))
    r = x
    for _ in range(config['n_dense']):
        r = convolutional_unit(r, config)
    r = tf.keras.layers.Flatten()(r)

    z_mean = tf.keras.layers.Dense(config['latent_dim'])(r)
    z_log_var = tf.keras.layers.Dense(config['latent_dim'], kernel_initializer='zero')(r)
    z = SampleNormal()([z_mean, z_log_var])

    # Decoder
    z_repeated = tf.keras.layers.RepeatVector(n_timesteps)(z)
    hd = z_repeated
    for _ in range(config['n_layers']):
        hd = recurrent_unit(z_repeated, config)

    # Estimate normal distribution for output features
    x_estimate = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(config['obs_per_timestep']))(hd)
    x_log_var = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(config['obs_per_timestep']))(hd)

    z_mean_m = tf.keras.models.Model(inputs=[x], outputs=[z_mean])
    z_log_var_m = tf.keras.models.Model(inputs=[x], outputs=[z_log_var])
    x_log_var_m = tf.keras.models.Model(inputs=[x], outputs=[x_log_var])

    model = tf.keras.models.Model(inputs=[x], outputs=[x_estimate])
    model.higgins_beta = tf.Variable(1e-5, trainable=False)

    def vae_loss(z_mean_m, z_log_var_m, x_log_var_m, higgins_beta=1e-5, epsilon=1e-10):

        def vae_loss_(x, x_decoded_mean):
            z_mean, z_log_var, x_log_var = z_mean_m(x), z_log_var_m(x), x_log_var_m(x)
            x_var = tf.keras.backend.exp(x_log_var) + epsilon
            l = 0.5 * tf.keras.backend.log(x_var) + (tf.keras.backend.square(x - x_decoded_mean)/(2.0 * x_var))
            xent_loss = tf.keras.backend.sum(tf.keras.backend.sum(l, axis=-1), axis=-1)
            kl_loss = -higgins_beta * tf.keras.backend.sum(1 + z_log_var - tf.keras.backend.square(z_mean) - tf.keras.backend.exp(z_log_var), axis=-1)
            return xent_loss + kl_loss

        return vae_loss_

    if 'metrics' in config:
        metrics = config['metrics']
    else:
        metrics = None

    optimizer = tf.keras.optimizers.Adam(lr=config['learning_rate'], beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=config['learning_rate_decay'])
    model.compile(loss=vae_loss(z_mean_m, z_log_var_m, x_log_var_m), optimizer=optimizer, metrics=metrics)

    encoder = tf.keras.models.Model(inputs=[x], outputs=[z_mean])
    latent_statistics = tf.keras.models.Model(inputs=[x], outputs=[z_mean, z_log_var])
    sampler = tf.keras.models.Model(inputs=[x], outputs=[x_estimate])

    return model, encoder, latent_statistics, sampler, True

def vae_conv2lstm_1D(config):
    # Create a simple variational autoencoder with a convolutional encoder and a lstm decoder inspired by PixelCNN https://arxiv.org/abs/1606.05328
    n_features = config['n_features']
    n_timesteps = int(n_features/config['obs_per_timestep'])
    shape = [None, n_features]

    # Encoder
    x = tf.keras.layers.Input(shape=(shape[1],))
    r = tf.keras.layers.Reshape((n_timesteps, config['obs_per_timestep']))(x)
    for _ in range(config['n_dense']):
        r = convolutional_unit(r, config)
    r = tf.keras.layers.Flatten()(r)

    z_mean = tf.keras.layers.Dense(config['latent_dim'])(r)
    z_log_var = tf.keras.layers.Dense(config['latent_dim'], kernel_initializer='zero')(r)
    z = SampleNormal()([z_mean, z_log_var])

    # Decoder
    z_repeated = tf.keras.layers.RepeatVector(n_timesteps)(z)
    hd = z_repeated
    for _ in range(config['n_layers']):
        hd = recurrent_unit(z_repeated, config)

    # Estimate normal distribution for output features
    x_estimate_t = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(config['obs_per_timestep']))(hd)
    x_log_var_t  = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(config['obs_per_timestep']))(hd)

    # Flatten the output
    x_estimate = tf.keras.layers.Flatten()(x_estimate_t)
    x_log_var = tf.keras.layers.Flatten()(x_log_var_t)

    z_mean_m = tf.keras.models.Model(inputs=[x], outputs=[z_mean])
    z_log_var_m = tf.keras.models.Model(inputs=[x], outputs=[z_log_var])
    x_log_var_m = tf.keras.models.Model(inputs=[x], outputs=[x_log_var])

    model = tf.keras.models.Model(x, x_estimate)

    def vae_loss(z_mean_m, z_log_var_m, x_log_var_m, higgins_beta=1e-5, epsilon=1e-10):
        def vae_loss_(x, x_decoded_mean):
            z_mean, z_log_var, x_log_var = z_mean_m(x), z_log_var_m(x), x_log_var_m(x)
            """VAE loss function"""
            x_var = tf.keras.backend.exp(x_log_var) + epsilon
            l = 0.5 * tf.keras.backend.log(x_var) + (tf.keras.backend.square(x - x_decoded_mean)/(2.0 * x_var))
            xent_loss = tf.keras.backend.sum(l, axis=-1)
            kl_loss = -higgins_beta * tf.keras.backend.sum(1 + z_log_var - tf.keras.backend.square(z_mean) - tf.keras.backend.exp(z_log_var), axis=-1)
            return xent_loss + kl_loss

        return vae_loss_

    if 'metrics' in config:
        metrics = config['metrics']
    else:
        metrics = None

    optimizer = tf.keras.optimizers.Adam(lr=config['learning_rate'], beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=config['learning_rate_decay'])
    model.compile(loss=vae_loss(z_mean_m, z_log_var_m, x_log_var_m), optimizer=optimizer, metrics=metrics)

    encoder = tf.keras.models.Model(inputs=[x], outputs=[z_mean])
    latent_statistics = tf.keras.models.Model(inputs=[x], outputs=[z_mean, z_log_var])
    sampler = tf.keras.models.Model(x, x_estimate)

    return model, encoder, latent_statistics, sampler, False

def vae_d2lstm(config):
    # Create a recurrent variational autoencoder
    n_features = config['n_features']
    n_timesteps = int(n_features/config['obs_per_timestep'])
    shape = [None, n_timesteps, config['obs_per_timestep']]

    # Encoder
    x = tf.keras.layers.Input(shape=(shape[1], shape[2]))
    r = tf.keras.layers.Flatten()(x)
    for _ in range(config['n_dense']):
        r = dense_unit(r, config)

    # Normal distribution in latent space
    z_mean = tf.keras.layers.Dense(config['latent_dim'])(r)
    z_log_var = tf.keras.layers.Dense(config['latent_dim'], kernel_initializer='zero')(r)
    z = SampleNormal()([z_mean, z_log_var])

    # Decoder
    z_repeated = tf.keras.layers.RepeatVector(n_timesteps)(z)
    hd = z_repeated
    for _ in range(config['n_layers']):
        hd = recurrent_unit(z_repeated, config)

    # Estimate normal distribution for output features
    x_estimate = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(config['obs_per_timestep']))(hd)
    x_log_var = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(config['obs_per_timestep']))(hd)

    z_mean_m = tf.keras.models.Model(inputs=[x], outputs=[z_mean])
    z_log_var_m = tf.keras.models.Model(inputs=[x], outputs=[z_log_var])
    x_log_var_m = tf.keras.models.Model(inputs=[x], outputs=[x_log_var])

    model = tf.keras.models.Model(x, x_estimate)

    def vae_loss(z_mean_m, z_log_var_m, x_log_var_m, higgins_beta=1e-5, epsilon=1e-10):

        def vae_loss_(x, x_decoded_mean):
            """VAE loss function"""
            z_mean, z_log_var, x_log_var = z_mean_m(x), z_log_var_m(x), x_log_var_m(x)
            x_var = tf.keras.backend.exp(x_log_var) + epsilon
            l = 0.5 * tf.keras.backend.log(x_var) + (tf.keras.backend.square(x - x_decoded_mean)/(2.0 * x_var))
            xent_loss = tf.keras.backend.sum(tf.keras.backend.sum(l, axis=-1), axis=-1)
            kl_loss = -higgins_beta * tf.keras.backend.sum(1 + z_log_var - tf.keras.backend.square(z_mean) - tf.keras.backend.exp(z_log_var), axis=-1)
            return xent_loss + kl_loss

        return vae_loss_

    if 'metrics' in config:
        metrics = config['metrics']
    else:
        metrics = None

    optimizer = tf.keras.optimizers.Adam(lr=config['learning_rate'], beta_1=0.9, beta_2=0.999,  epsilon=1e-08, decay=config['learning_rate_decay'])
    model.compile(loss=vae_loss(z_mean_m, z_log_var_m, x_log_var_m), optimizer=optimizer, metrics=metrics)

    encoder = tf.keras.models.Model(inputs=[x], outputs=[z_mean])
    latent_statistics = tf.keras.models.Model(inputs=[x], outputs=[z_mean, z_log_var])
    sampler = tf.keras.models.Model(x, x_estimate)

    return model, encoder, latent_statistics, sampler, True

def vae_d2lstm_1D(config):
    # Create a recurrent variational autoencoder with 1D input/output adapted from the d2lstm network above
    n_features = config['n_features']
    n_timesteps = int(n_features/config['obs_per_timestep'])
    shape = [None, n_features]

    # Encoder
    x = tf.keras.layers.Input(shape=(shape[1],))
    r = x
    for _ in range(config['n_dense']):
        r = dense_unit(r, config)

    # Normal distribution in latent space
    z_mean = tf.keras.layers.Dense(config['latent_dim'])(r)
    z_log_var = tf.keras.layers.Dense(config['latent_dim'], kernel_initializer='zero')(r)
    z = SampleNormal()([z_mean, z_log_var])

    # Decoder
    z_repeated = tf.keras.layers.RepeatVector(n_timesteps)(z)
    hd = recurrent_unit(z_repeated, config)

    # Estimate normal distribution for output features
    x_estimate_t = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(config['obs_per_timestep']))(hd)
    x_log_var_t = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(config['obs_per_timestep']))(hd)

    # Flatten the output
    x_estimate = tf.keras.layers.Flatten()(x_estimate_t)
    x_log_var = tf.keras.layers.Flatten()(x_log_var_t)

    z_mean_m = tf.keras.models.Model(inputs=[x], outputs=[z_mean])
    z_log_var_m = tf.keras.models.Model(inputs=[x], outputs=[z_log_var])
    x_log_var_m = tf.keras.models.Model(inputs=[x], outputs=[x_log_var])

    model = tf.keras.models.Model(x, x_estimate)

    def vae_loss(z_mean_m, z_log_var_m, x_log_var_m, higgins_beta=1e-5, epsilon=1e-10):
        def vae_loss_(x, x_decoded_mean):
            """VAE loss function"""
            z_mean, z_log_var, x_log_var = z_mean_m(x), z_log_var_m(x), x_log_var_m(x)
            x_var = tf.keras.backend.exp(x_log_var) + epsilon
            l = 0.5 * tf.keras.backend.log(x_var) + (tf.keras.backend.square(x - x_decoded_mean)/(2.0 * x_var))
            xent_loss = tf.keras.backend.sum(l, axis=-1)
            kl_loss = -higgins_beta * tf.keras.backend.sum(1 + z_log_var - tf.keras.backend.square(z_mean) - tf.keras.backend.exp(z_log_var), axis=-1)
            return xent_loss + kl_loss

        return vae_loss_

    optimizer = tf.keras.optimizers.Adam(lr=config['learning_rate'], beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=config['learning_rate_decay'])
    model.compile(loss=vae_loss(z_mean_m, z_log_var_m, x_log_var_m), optimizer=optimizer)

    encoder = tf.keras.models.Model(inputs=[x], outputs=[z_mean])
    latent_statistics = tf.keras.models.Model(inputs=[x], outputs=[z_mean, z_log_var])
    sampler = tf.keras.models.Model(x, x_estimate)

    return model, encoder, latent_statistics, sampler, False

def variational_simple(config):
    # Create a simple non-recurrent variational autoencoder
    n_features = config['n_features']
    shape = [None, n_features]

    # Encoder
    x = tf.keras.layers.Input(shape=(shape[1],))
    r = x
    for _ in range(config['n_dense']):
        r = dense_unit(r, config)

    # Normal distribution in latent space
    z_mean = tf.keras.layers.Dense(config['latent_dim'])(r)
    z_log_var = tf.keras.layers.Dense(config['latent_dim'], kernel_initializer='zero')(r)
    z = SampleNormal()([z_mean, z_log_var])

    # Decoder
    h = z
    for _ in range(config['n_dense']):
        h = dense_unit(h, config)

    # Estimate normal distribution for output features
    x_estimate = tf.keras.layers.Dense(n_features)(h)
    x_log_var = tf.keras.layers.Dense(n_features)(h)

    z_mean_m = tf.keras.models.Model(inputs=[x], outputs=[z_mean])
    z_log_var_m = tf.keras.models.Model(inputs=[x], outputs=[z_log_var])
    x_log_var_m = tf.keras.models.Model(inputs=[x], outputs=[x_log_var])

    model = tf.keras.models.Model(x, x_estimate)

    def vae_loss(z_mean_m, z_log_var_m, x_log_var_m, higgins_beta=1e-5, epsilon=1e-10):
        def vae_loss_(x, x_decoded_mean):
            """VAE loss function"""
            z_mean, z_log_var, x_log_var = z_mean_m(x), z_log_var_m(x), x_log_var_m(x)
            x_var = tf.keras.backend.exp(x_log_var) + epsilon
            l = 0.5 * tf.keras.backend.log(x_var) + (tf.keras.backend.square(x - x_decoded_mean)/(2.0 * x_var))
            xent_loss = tf.keras.backend.mean(l, axis=-1)
            kl_loss = -higgins_beta * tf.keras.backend.mean(1 + z_log_var - tf.keras.backend.square(z_mean) - tf.keras.backend.exp(z_log_var), axis=-1)
            return xent_loss + kl_loss

        return vae_loss_

    if 'metrics' in config:
        metrics = config['metrics']
    else:
        metrics = None

    optimizer = tf.keras.optimizers.Adam(lr=config['learning_rate'], beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=config['learning_rate_decay'])
    model.compile(loss=vae_loss(z_mean_m, z_log_var_m, x_log_var_m), optimizer=optimizer, metrics=metrics)

    encoder = tf.keras.models.Model(inputs=[x], outputs=[z_mean])
    latent_statistics = tf.keras.models.Model(inputs=[x], outputs=[z_mean, z_log_var])
    sampler = tf.keras.models.Model(x, x_estimate)

    return model, encoder, latent_statistics, sampler, False

def direct_simple(config):
    # Create a simple non-recurrent autoencoder
    n_features = config['n_features']
    shape = [None, n_features]

    # Encoder
    x = tf.keras.layers.Input(shape=(shape[1],))
    r = x
    for _ in range(config['n_dense']):
        r = dense_unit(r, config)

    # Normal distribution in latent space
    z = tf.keras.layers.Dense(config['latent_dim'])(r)

    # Decoder
    h = z
    for _ in range(config['n_dense']):
        h = dense_unit(h, config)

    # Estimate normal distribution for output features
    x_estimate = tf.keras.layers.Dense(n_features)(h)

    model = tf.keras.models.Model(x, x_estimate)
    model.higgins_beta = tf.keras.layers.Input(tensor=tf.constant(1e-5))

    optimizer = tf.keras.optimizers.Adam(lr=config['learning_rate'], beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=config['learning_rate_decay'])
    model.compile(loss="mse", optimizer=optimizer)

    encoder = tf.keras.models.Model(inputs=[x], outputs=[z])
    latent_statistics = tf.keras.models.Model(inputs=[x], outputs=[z, z])
    sampler = tf.keras.models.Model(x, x_estimate)

    return model, encoder, latent_statistics, sampler, False

def variational(config):
    # Create a variational autoencoder with similar to Kingma14 or Higgins16
    n_features = config['n_features']
    n_timesteps = int(n_features/config['obs_per_timestep'])
    shape = [None, n_timesteps, config['obs_per_timestep']]
    rnn = tf.keras.layers.LSTM

    # Encoder
    x = tf.keras.layers.Input(shape=(shape[1], shape[2]))
    h = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(config['inner_dim']), input_shape=(shape[1], shape[2]))(x)
    h = tf.keras.layers.BatchNormalization()(h)
    for _ in range(config['n_layers']):
        h = rnn(config['inner_dim'], return_sequences=True)(h)
        h = tf.keras.layers.BatchNormalization()(h)
    r = rnn(config['inner_dim'], input_shape=(None, config['inner_dim']))(h)

    r = tf.keras.layers.BatchNormalization()(r)
    for _ in range(config['n_dense']):
        r = tf.keras.layers.Dense(config['inner_dim'], activation='relu')(r)
        r = tf.keras.layers.BatchNormalization()(r)

    z_mean = tf.keras.layers.Dense(config['latent_dim'])(r)
    z_log_var = tf.keras.layers.Dense(config['latent_dim'])(r)
    z = SampleNormal()([z_mean, z_log_var])

    hd = z
    for _ in range(config['n_dense']):
        hd = tf.keras.layers.Dense(config['inner_dim'], activation='relu')(hd)
        hd = tf.keras.layers.BatchNormalization()(hd)

    # Decoder
    z_repeated = tf.keras.layers.RepeatVector(shape[1])(hd)
    for i in range(config['n_layers']):
        z_repeated = rnn(config['inner_dim'], return_sequences=True)(z_repeated)
        z_repeated = tf.keras.layers.BatchNormalization()(z_repeated)
    d = rnn(config['inner_dim'], return_sequences=True)(z_repeated)
    d = tf.keras.layers.BatchNormalization()(d)
    x_estimate = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(config['output_dim']))(d)
    x_log_var = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(config['output_dim']))(d)

    z_mean_m = tf.keras.models.Model(inputs=[x], outputs=[z_mean])
    z_log_var_m = tf.keras.models.Model(inputs=[x], outputs=[z_log_var])
    x_log_var_m = tf.keras.models.Model(inputs=[x], outputs=[x_log_var])

    def vae_loss(z_mean_m, z_log_var_m, x_log_var_m, epsilon=1e-10):
        def vae_loss_(x, x_decoded_mean):
            z_mean, z_log_var, x_log_var = z_mean_m(x), z_log_var_m(x), x_log_var_m(x)
            x_sigma = tf.keras.backend.exp(x_log_var) + epsilon

            l = 0.5 * tf.keras.backend.log(x_sigma) + (tf.keras.backend.square(x - x_decoded_mean)/(2.0 * x_sigma))
            xent_loss = tf.keras.backend.sum(tf.keras.backend.sum(l, axis=-1), axis=-1)

            kl_loss = -config['beta'] * tf.keras.backend.sum(1 + z_log_var - tf.keras.backend.square(z_mean) - tf.keras.backend.exp(z_log_var), axis=-1)

            return xent_loss + kl_loss

        return vae_loss_

    model = tf.keras.models.Model(x, x_estimate)
    optimizer = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=config['learning_rate_decay'])

    model.compile(loss=vae_loss(z_mean_m, z_log_var_m, x_log_var_m), optimizer=optimizer)

    encoder = tf.keras.models.Model(inputs=[x], outputs=[z_mean])
    latent_statistics = tf.keras.models.Model(inputs=[x], outputs=[z_mean, z_log_var])
    sampler = tf.keras.models.Model(x, x_estimate)

    return model, encoder, latent_statistics, sampler, True

def direct(config):
    n_features = config['n_features']
    n_timesteps = int(n_features/config['obs_per_timestep'])
    shape = [None, n_timesteps, config['obs_per_timestep']]
    rnn = tf.keras.layers.LSTM

    # Encoder
    x = tf.keras.layers.Input(shape=(shape[1], shape[2]))
    h = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(config['inner_dim']), input_shape=(shape[1], shape[2]))(x)
    for i in range(config['n_layers']):
        h = rnn(config['inner_dim'], return_sequences=True)(h)
    r = rnn(config['inner_dim'], input_shape=(None, config['inner_dim']))(h)
    z = tf.keras.layers.Dense(config['latent_dim'])(r)

    # Decoder
    z_repeated = tf.keras.layers.RepeatVector(shape[1])(z)
    for i in range(config['n_layers']):
        z_repeated = rnn(config['inner_dim'], return_sequences=True)(z_repeated)
    d = rnn(config['inner_dim'], return_sequences=True)(z_repeated)
    x_estimate = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(config['output_dim']))(d)

    model = tf.keras.models.Model(x, x_estimate)
    optimizer = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    encoder = tf.keras.models.Model(inputs=[x], outputs=[z])
    end_to_end = tf.keras.models.Model(inputs=[x], outputs=[x_estimate])

    return model, encoder, end_to_end

def l1norm(config):
    n_features = config['n_features']
    n_timesteps = int(n_features/config['obs_per_timestep'])
    shape = [None, n_timesteps, config['obs_per_timestep']]
    rnn = tf.keras.layers.LSTM

    # Encoder
    x = tf.keras.layers.Input(shape=(shape[1], shape[2]))
    h = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(config['inner_dim']), input_shape=(shape[1], shape[2]))(x)
    r = rnn(config['inner_dim'], input_shape=(None, config['inner_dim']))(h)
    z = tf.keras.layers.Dense(config['latent_dim'])(r)

    # Decoder
    z_repeated = tf.keras.layers.RepeatVector(shape[1])(z)
    d = rnn(config['inner_dim'], return_sequences=True)(z_repeated)
    x_estimate = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(config['output_dim']))(d)

    z_m = tf.keras.models.Model(inputs=[x], outputs=[z])

    def vae_loss(z_m):
        def vae_loss_(x, x_decoded_mean):
            z = z_m(x)
            xent_loss = tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.losses.mean_squared_error(x, x_decoded_mean), axis=-1))
            kl_loss = tf.keras.backend.mean(tf.keras.backend.abs(z), axis=-1)
            return xent_loss + kl_loss

        return vae_loss_

    model = tf.keras.models.Model(x, x_estimate)
    optimizer = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss=vae_loss(z_m), optimizer=optimizer)

    encoder = tf.keras.models.Model(inputs=[x], outputs=[z])
    end_to_end = tf.keras.models.Model(inputs=[x], outputs=[x_estimate])

    return model, encoder, end_to_end

def corr(config):
    n_features = config['n_features']
    n_timesteps = int(n_features/config['obs_per_timestep'])
    shape = [None, n_timesteps, config['obs_per_timestep']]
    rnn = tf.keras.layers.LSTM

    # Encoder
    x = tf.keras.layers.Input(shape=(shape[1], shape[2]))
    h = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(config['inner_dim']), input_shape=(shape[1], shape[2]))(x)
    for i in range(config['n_layers']):
        h = rnn(config['inner_dim'], return_sequences=True)(h)
    encoded = rnn(config['inner_dim'], input_shape=(None, config['inner_dim']))(h)
    z = tf.keras.layers.Dense(config['latent_dim'])(encoded)

    # Decoder
    z_repeated = tf.keras.layers.RepeatVector(shape[1])(z)
    for i in range(config['n_layers']):
        z_repeated = rnn(config['inner_dim'], return_sequences=True)(z_repeated)
    decoded = rnn(config['inner_dim'], return_sequences=True)(z_repeated)
    x_estimate = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(config['output_dim']))(decoded)

    z_m = tf.keras.models.Model(inputs=[x], outputs=[z])

    def corr_loss(z):
        # Computes the average pairwise linear rank correlation (Pearson) between the variables in z (columns)
        m = tf.keras.backend.mean(z, 0, keepdims=True)
        t = z-m
        coef = tf.matmul(tf.keras.backend.transpose(t), t)
        d = tf.sqrt(tf.linalg.diag_part(coef))
        n_samples = tf.keras.backend.shape(d)[0]
        d = tf.reshape(d, shape=(1, n_samples))
        tmp = tf.matmul(tf.keras.backend.transpose(d), d)
        corr = tf.truediv(coef, tmp)
        return tf.keras.backend.mean(corr)

    def total_loss(z_m):

        def total_loss_(x, x_decoded_mean):
            # Linear combination of mse and average correlation between dimensions in latent space"""
            z = z_m(x)
            xent_loss = tf.keras.backend.mean(tf.keras.losses.mean_squared_error(x, x_decoded_mean), axis=-1)
            corr = corr_loss(z)
            return xent_loss + corr

        return total_loss_

    model = tf.keras.models.Model(x, x_estimate)
    optimizer = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss=total_loss(z_m), optimizer=optimizer)

    encoder = tf.keras.models.Model(inputs=[x], outputs=[z])
    end_to_end = tf.keras.models.Model(inputs=[x], outputs=[x_estimate])

    return model, encoder, end_to_end
