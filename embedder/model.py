from tensorflow.keras.layers import Input, Embedding, Dense, Layer
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.regularizers import l1
import tensorflow as tf
import numpy as np


class WeightedAttention(Layer):
    def __init__(self, hidden_dim, n_heads):
        super(WeightedAttention, self).__init__()
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
    
    def build(self, input_shape):
        dim = input_shape[-1]

        initializer = RandomNormal(mean=0., stddev=1.)
        self.query = self.add_weight(
            "query", (self.n_heads, dim, 1),
            initializer=initializer,
            regularizer=l1(1e-5),
            dtype=np.float32)
        self.values = self.add_weight(
            "values", (self.n_heads, dim, self.hidden_dim),
            initializer=initializer,
            regularizer=l1(1e-5),
            dtype=np.float32)
        self.W = self.add_weight(
            "W", (self.n_heads * self.hidden_dim, dim),
            initializer=initializer,
            dtype=np.float32)

    def call(self, input, mask=None):
        if mask is not None:
            mask = tf.cast(mask, tf.float32)[:, :, None]
        input *= mask

        results = []
        for i in range(self.n_heads):
            query = tf.matmul(input, self.query[i])
            score = tf.nn.softmax(query, axis=-2)

            values = tf.matmul(input, self.values[i])
            results.append(values * score)

        results = tf.concat(results, axis=-1)
        results = tf.nn.tanh(tf.math.reduce_sum(results, axis=-2, keepdims=False))

        results = tf.nn.tanh(tf.matmul(results, self.W))
        results = tf.math.l2_normalize(results, axis=-1)
        return results


class Cosine(Layer):
    def __init__(self):
        super(Cosine, self).__init__()
    
    def call(self, inputs):
        left, right = inputs
        res = tf.math.reduce_sum(left * right, axis=-1, keepdims=True)
        return res


def create_model(n_features, n_heads, in_dim, hidden_dim):
    inp_left = Input(shape=(None,))
    inp_right = Input(shape=(None,))

    embedding = Embedding(n_features + 1, in_dim, mask_zero=True)
    weighted_attention = WeightedAttention(hidden_dim, n_heads)
    
    encoder = Sequential(name="encoder")
    encoder.add(embedding)
    encoder.add(weighted_attention)

    left = encoder(inp_left)
    right = encoder(inp_right)

    cosine = Cosine()([left, right])
    out = Dense(1, activation="sigmoid")(cosine)

    model = Model([inp_left, inp_right], out)
    model.compile("nadam", "binary_crossentropy", metrics=["accuracy"])

    encoder_inp = Input(shape=(None,))
    encoder_out = encoder(encoder_inp)
    encoder_model = Model(encoder_inp, encoder_out)
    encoder_model.compile("nadam", "binary_crossentropy", metrics=["accuracy"])
    return model, encoder_model
