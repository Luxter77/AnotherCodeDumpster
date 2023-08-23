import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from tensorflow.keras.layers import Dense, Dropout, LayerNormalization
import matplotlib.pyplot as plt
from typing import List, Tuple
from tqdm.auto import tqdm
from typing import Tuple
import tensorflow as tf
import pandas as pd
import pprint as pp
import numpy as np
import time


def get_angles(position: np.ndarray, i: np.ndarray, d_model: int) -> np.ndarray:
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return position * angle_rates


def create_padding_mask(seq: tf.Tensor) -> tf.Tensor:
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(size: int) -> tf.Tensor:
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask


def scaled_dot_product_attention(q: tf.Tensor, k: tf.Tensor, v: tf.Tensor, mask: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    d_k = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(d_k)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)
    return output, attention_weights


def positional_encoding(position: int, d_model: int) -> tf.Tensor:
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)


def point_wise_feed_forward_network(d_model: int, dff: int) -> tf.keras.Sequential:
    return tf.keras.Sequential([
        Dense(dff, activation='relu'),
        Dense(d_model)
    ])


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model: int, num_heads: int):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        self.wq = Dense(d_model)
        self.wk = Dense(d_model)
        self.wv = Dense(d_model)
        self.dense = Dense(d_model)

    def split_heads(self, x: tf.Tensor, batch_size: int) -> tf.Tensor:
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v: tf.Tensor, k: tf.Tensor, q: tf.Tensor, mask: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        batch_size = tf.shape(q)[0]
        q = self.split_heads(self.wq(q), batch_size)
        k = self.split_heads(self.wk(k), batch_size)
        v = self.split_heads(self.wv(v), batch_size)

        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(
            scaled_attention, (batch_size, -1, self.d_model))
        output = self.dense(concat_attention)
        return output, attention_weights


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model: int, num_heads: int, dff: int, dropout_rate: float):
        super(EncoderLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def call(self, x: tf.Tensor, training: bool, mask: tf.Tensor) -> tf.Tensor:
        attn_output, _ = self.multi_head_attention(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers: int, d_model: int, num_heads: int, dff: int, input_vocab_size: int,
                 maximum_position_encoding: int, dropout_rate: float, cnn_filter_size: int, cnn_num_filters: int):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.cnn = tf.keras.layers.Conv1D(filters=cnn_num_filters, kernel_size=cnn_filter_size, padding='same', activation='relu')
        self.pos_encoding = positional_encoding(
            maximum_position_encoding, self.d_model)
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, dropout_rate)
                           for _ in range(num_layers)]
        self.dropout = Dropout(dropout_rate)

    def call(self, x: tf.Tensor, training: bool, mask: tf.Tensor) -> tf.Tensor:
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        
        # Apply the CNN layer to capture local patterns
        x = self.cnn(x)
        
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
        return x



class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model: int, num_heads: int, dff: int, dropout_rate: float):
        super(DecoderLayer, self).__init__()
        self.multi_head_attention1 = MultiHeadAttention(d_model, num_heads)
        self.multi_head_attention2 = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.layernorm3 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
        self.dropout3 = Dropout(dropout_rate)

    def call(self, x: tf.Tensor, enc_output: tf.Tensor, training: bool, look_ahead_mask: tf.Tensor, padding_mask: tf.Tensor) -> tf.Tensor:
        attn1, attn_weights_block1 = self.multi_head_attention1(
            x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.multi_head_attention2(
            enc_output, enc_output, out1, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)
        return out3


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers: int, d_model: int, num_heads: int, dff: int, target_vocab_size: int,
                 maximum_position_encoding: int, dropout_rate: float):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(
            maximum_position_encoding, self.d_model)
        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, dropout_rate)
                           for _ in range(num_layers)]
        self.dropout = Dropout(dropout_rate)

    def call(self, x: tf.Tensor, enc_output: tf.Tensor, training: bool, look_ahead_mask: tf.Tensor, padding_mask: tf.Tensor) -> tf.Tensor:
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.dec_layers[i](
                x, enc_output, training, look_ahead_mask, padding_mask)
        return x


class Transformer(tf.keras.Model):
    def __init__(self, num_layers: int, d_model: int, num_heads: int, dff: int, input_vocab_size: int,
                 target_vocab_size: int, pe_input: int, pe_target: int, dropout_rate: float):
        super(Transformer, self).__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size,
                               pe_input, dropout_rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size,
                               pe_target, dropout_rate)
        self.final_layer = Dense(target_vocab_size)

    def call(self, inp: tf.Tensor, tar: tf.Tensor, training: bool, enc_padding_mask: tf.Tensor,
             look_ahead_mask: tf.Tensor, dec_padding_mask: tf.Tensor) -> tf.Tensor:
        enc_output = self.encoder(inp, training, enc_padding_mask)
        dec_output = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask)
        final_output = self.final_layer(dec_output)
        return final_output

# Hyperparameters
NUM_LAYERS = 6
D_MODEL = 512
NUM_HEADS = 8
DFF = 2048
INPUT_VOCAB_SIZE = 10000
TARGET_VOCAB_SIZE = 8000
DROPOUT_RATE = 0.1
MAXIMUM_POSITION_ENCODING = 10000
TARGET_SEQ_LENGTH = 50

if __name__ == "__main__":
    # Replace "your_data.csv" with the actual CSV filename
    data = pd.read_csv("your_data.csv")
    sample_input = data["input_column_name"].to_numpy(
        dtype=np.int64)
    sample_target = data["target_column_name"].to_numpy(
        dtype=np.int64)

    # Create masks
    enc_padding_mask = create_padding_mask(sample_input)
    look_ahead_mask = create_look_ahead_mask(TARGET_SEQ_LENGTH)
    dec_padding_mask = create_padding_mask(sample_target)

    # Apply masks
    look_ahead_mask = tf.where(look_ahead_mask == 0, -np.inf, look_ahead_mask)
    dec_padding_mask = tf.where(dec_padding_mask == 0, -np.inf, dec_padding_mask)

    # Create the Transformer model
    sample_transformer = Transformer(NUM_LAYERS, D_MODEL, NUM_HEADS, DFF,
                                    INPUT_VOCAB_SIZE, TARGET_VOCAB_SIZE,
                                    pe_input=MAXIMUM_POSITION_ENCODING,
                                    pe_target=TARGET_SEQ_LENGTH,
                                    dropout_rate=DROPOUT_RATE)

    # Test the model
    sample_output = sample_transformer(sample_input, sample_target, training=False,
                                    enc_padding_mask=enc_padding_mask,
                                    look_ahead_mask=look_ahead_mask,
                                    dec_padding_mask=dec_padding_mask)
