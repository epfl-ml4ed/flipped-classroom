#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf

class Attention(tf.keras.layers.Layer):

    def __init__(self, W_regularizer=None, b_regularizer=None, W_constraint=None, b_constraint=None, bias=True, return_attention=False, **kwargs):

        self.supports_masking = True
        self.return_attention = return_attention
        self.init = tf.keras.initializers.get('glorot_uniform')

        self.W_regularizer = tf.keras.regularizers.get(W_regularizer)
        self.b_regularizer = tf.keras.regularizers.get(b_regularizer)

        self.W_constraint = tf.keras.constraints.get(W_constraint)
        self.b_constraint = tf.keras.constraints.get(b_constraint)

        self.bias = bias
        super(Attention, self).__init__(**kwargs)

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'supports_masking': self.supports_masking,
            'return_attention': self.return_attention,
            'init': self.init,
            'W_regularizer': self.W_regularizer,
            'b_regularizer': self.b_regularizer,
            'W_constraint': self.W_constraint,
            'b_constraint': self.b_constraint,
            'bias': self.bias
        })
        return config

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1],), initializer=self.init, name='{}_W'.format(self.name), regularizer=self.W_regularizer, constraint=self.W_constraint)

        if self.bias:
            self.b = self.add_weight(shape=(input_shape[1],), initializer='zero', name='{}_b'.format(self.name), regularizer=self.b_regularizer, constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        eij = tf.keras.backend.squeeze(tf.keras.backend.dot(x, tf.keras.backend.expand_dims(self.W)), axis=-1)

        if self.bias:
            eij += self.b

        eij = tf.keras.backend.tanh(eij)

        a = tf.keras.backend.exp(eij)

        if mask is not None:
            a *= tf.keras.backend.cast(mask, tf.keras.backend.floatx())

        a /= tf.keras.backend.cast(tf.keras.backend.sum(a, axis=1, keepdims=True) + tf.keras.backend.epsilon(), tf.keras.backend.floatx())

        weighted_input = x * tf.keras.backend.expand_dims(a)

        result = tf.keras.backend.sum(weighted_input, axis=1)

        if self.return_attention:
            return [result, a]
        return result

    def compute_output_shape(self, input_shape):
        if self.return_attention:
            return [(input_shape[0], input_shape[-1]),
                    (input_shape[0], input_shape[1])]
        else:
            return input_shape[0], input_shape[-1]