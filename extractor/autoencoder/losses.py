#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf

def corr_loss(z):
    m = tf.keras.backend.mean(z, 0, keepdims=True)
    t = z - m
    coef = tf.matmul(tf.keras.backend.transpose(t), t)
    d = tf.sqrt(tf.linalg.diag_part(coef))
    n_samples = tf.keras.backend.shape(d)[0]
    d = tf.reshape(d, shape=(1, n_samples))
    tmp = tf.matmul(tf.keras.backend.transpose(d), d)
    corr = tf.truediv(coef, tmp)
    return tf.keras.backend.mean(corr)

def total_loss(z_m):
    def total_loss_(x, x_decoded_mean):
        z = z_m(x)
        xent_loss = tf.keras.backend.mean(tf.keras.losses.mean_squared_error(x, x_decoded_mean), axis=-1)
        corr = corr_loss(z)
        return xent_loss + corr
    return total_loss_

def vae_loss(z_m):
    def vae_loss_(x, x_decoded_mean):
        z = z_m(x)
        xent_loss = tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.losses.mean_squared_error(x, x_decoded_mean), axis=-1))
        kl_loss = tf.keras.backend.mean(tf.keras.backend.abs(z), axis=-1)
        return xent_loss + kl_loss
    return vae_loss_

def vae_loss_basic(z_mean_m, z_log_var_m, x_log_var_m, beta=0.5, epsilon=1e-10):
    def vae_loss_(x, x_decoded_mean):
        z_mean, z_log_var, x_log_var = z_mean_m(x), z_log_var_m(x), x_log_var_m(x)
        x_sigma = tf.keras.backend.exp(x_log_var) + epsilon
        l = 0.5 * tf.keras.backend.log(x_sigma) + (tf.keras.backend.square(x - x_decoded_mean) / (2.0 * x_sigma))
        xent_loss = tf.keras.backend.sum(tf.keras.backend.sum(l, axis=-1), axis=-1)
        kl_loss = -beta * tf.keras.backend.sum(1 + z_log_var - tf.keras.backend.square(z_mean) - tf.keras.backend.exp(z_log_var), axis=-1)
        return xent_loss + kl_loss
    return vae_loss_