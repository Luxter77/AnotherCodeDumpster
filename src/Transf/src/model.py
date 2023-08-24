from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization


def get_angles(position: np.ndarray, i: np.ndarray, d_model: int) -> np.ndarray:
    """
    Calculate the angles for positional encodings in a Transformer model.

    This function computes the angle rates for the positional encodings used in
    the Transformer model. The positional encodings are added to the input
    embeddings to provide information about the order of tokens in the sequence.

    Args:
        position (np.ndarray): An array representing the positions of tokens in a sequence.
        i (np.ndarray): An array containing the indices of the positions.
        d_model (int): The dimension of the model.

    Returns:
        np.ndarray: An array containing the angles for positional encodings.

    Notes:
        - The formula for calculating angle rates is: 1 / (10000 ** (2 * (i // 2) / d_model))
        - This function is often used in the context of the Transformer's positional encoding.
          The positional encodings help the model take into account the order of tokens
          in the sequence, which is crucial for tasks involving sequential data.

    Example:
        >>> position = np.array([[0, 1, 2], [3, 4, 5]])
        >>> i = np.array([[0, 1, 2], [0, 1, 2]])
        >>> d_model = 512
        >>> angles = get_angles(position, i, d_model)
        >>> print(angles)
        [[0.00000000e+00 1.00000000e+00 1.00000000e+00]
         [0.00000000e+00 6.30957344e-01 6.30957344e-01]]

    """
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return position * angle_rates


def create_padding_mask(seq: tf.Tensor) -> tf.Tensor:
    """
    Create a padding mask for a sequence.

    Args:
        seq (tf.Tensor): Input sequence tensor.

    Returns:
        tf.Tensor: Padding mask tensor with shape (batch_size, 1, 1, seq_length).
        
    This function takes an input sequence tensor and creates a padding mask to identify
    the positions of padding elements in the sequence. The padding mask is a binary tensor
    where each element is 1 if the corresponding element in the input sequence is a padding
    element (value 0), and 0 otherwise. The mask is expanded with extra dimensions to
    make it compatible with subsequent operations.
    """
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(size: int) -> tf.Tensor:
    """
    Create a look-ahead mask for self-attention mechanisms in sequence-to-sequence tasks.

    This function generates a mask that prevents each position from attending to subsequent positions
    in a sequence. It's particularly useful in scenarios where an autoregressive model needs to
    generate sequences step by step without accessing future information.

    Args:
        size (int): The size of the mask, typically equal to the sequence length.

    Returns:
        tf.Tensor: A lower-triangular matrix with ones in the lower triangle and zeros in the upper triangle,
        preventing information flow from future positions during self-attention computations.
    """
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask


def scaled_dot_product_attention(q: tf.Tensor, k: tf.Tensor, v: tf.Tensor, mask: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Calculate scaled dot-product attention.

    Args:
        q (tf.Tensor): Query tensor with shape (..., seq_len_q, depth).
        k (tf.Tensor): Key tensor with shape (..., seq_len_k, depth).
        v (tf.Tensor): Value tensor with shape (..., seq_len_v, depth_v).
        mask (tf.Tensor): Mask tensor with shape (..., seq_len_q, seq_len_k), containing 0s and 1s,
                         where 1s indicate positions to be masked.

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: A tuple containing:
            - output (tf.Tensor): Output tensor after attention mechanism with shape (..., seq_len_q, depth_v).
            - attention_weights (tf.Tensor): Attention weights tensor with shape (..., seq_len_q, seq_len_k).
    """
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    d_k = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(d_k)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)
    return output, attention_weights


def positional_encoding(position: int, d_model: int) -> tf.Tensor:
    """
    Generates positional encodings for a given sequence of positions and dimensions.

    Args:
        position (int): The maximum position index.
        d_model (int): The dimension of the model/embedding.

    Returns:
        tf.Tensor: A tensor containing the positional encodings.

    The positional encoding is used to provide information about the order of elements in a sequence
    to the model. This is important for models like transformers that do not inherently understand
    the sequential order of inputs.

    The encoding is generated using sinusoidal functions of different frequencies. Each dimension of the
    encoding corresponds to a sinusoid of a different frequency.

    The even dimensions of the encoding are filled with the sine of the computed angles, while the odd
    dimensions are filled with the cosine of the angles. This alternating pattern ensures that adjacent
    dimensions have distinct encoding values.

    The final positional encoding is returned as a tensor with shape (1, position, d_model), suitable for
    adding to the input embeddings of a sequence.
    """
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)


def point_wise_feed_forward_network(d_model: int, dff: int) -> tf.keras.Sequential:
    """
    Construct a point-wise feed-forward neural network.

    This function creates a feed-forward network composed of two dense (fully connected) layers.
    
    Args:
        d_model (int): The dimensionality of the model's output space.
        dff (int): The number of units in the hidden (intermediate) layer.

    Returns:
        tf.keras.Sequential: A sequential model representing the point-wise feed-forward network.
    """
    return tf.keras.Sequential([
        Dense(dff, activation='relu'),
        Dense(d_model)
    ])


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Implementation of the Multi-Head Attention mechanism layer.

    Args:
        d_model (int): The dimensionality of the model.
        num_heads (int): The number of attention heads.

    Attributes:
        num_heads (int): The number of attention heads.
        d_model (int): The dimensionality of the model.
        depth (int): The depth of the model, calculated as d_model // num_heads.
        wq (Dense): Linear transformation layer for queries.
        wk (Dense): Linear transformation layer for keys.
        wv (Dense): Linear transformation layer for values.
        dense (Dense): Linear transformation layer for the final output.

    Methods:
        split_heads(x, batch_size): Splits the input tensor into multiple heads.
        call(v, k, q, mask): Executes the multi-head attention mechanism.

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: The output tensor and attention weights.

    Example:
        # Create a MultiHeadAttention instance
        multihead_attn = MultiHeadAttention(d_model=512, num_heads=8)
        
        # Generate some sample input tensors
        v = tf.random.uniform(shape=(batch_size, sequence_length, d_model))
        k = tf.random.uniform(shape=(batch_size, sequence_length, d_model))
        q = tf.random.uniform(shape=(batch_size, sequence_length, d_model))
        mask = create_padding_mask(input_sequence)  # Replace with actual mask

        # Obtain the output tensor and attention weights
        output, attention_weights = multihead_attn(v, k, q, mask)
    """
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
        """
        Splits the input tensor into multiple heads.

        Args:
            x (tf.Tensor): The input tensor.
            batch_size (int): The batch size.

        Returns:
            tf.Tensor: The tensor after splitting.
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v: tf.Tensor, k: tf.Tensor, q: tf.Tensor, mask: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Executes the multi-head attention mechanism.

        Args:
            v (tf.Tensor): The value tensor.
            k (tf.Tensor): The key tensor.
            q (tf.Tensor): The query tensor.
            mask (tf.Tensor): The mask tensor.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: The output tensor and attention weights.
        """
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
    """
    A single layer of the Transformer encoder.

    Args:
        d_model (int): The dimensionality of the model's hidden representations.
        num_heads (int): The number of attention heads in the multi-head attention mechanism.
        dff (int): The number of units in the feed-forward neural network.
        dropout_rate (float): The dropout rate applied to the output of each sub-layer.

    Attributes:
        multi_head_attention (MultiHeadAttention): Multi-head self-attention mechanism.
        ffn (tf.keras.Sequential): Point-wise feed-forward neural network.
        layernorm1 (LayerNormalization): Layer normalization after the first sub-layer.
        layernorm2 (LayerNormalization): Layer normalization after the second sub-layer.
        dropout1 (Dropout): Dropout layer after the first sub-layer.
        dropout2 (Dropout): Dropout layer after the second sub-layer.

    Methods:
        call(x, training, mask):
            Applies the encoder layer to the input tensor.
        
    Returns:
        tf.Tensor: The output tensor after passing through the encoder layer.
    """
    def __init__(self, d_model: int, num_heads: int, dff: int, dropout_rate: float):
        super(EncoderLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def call(self, x: tf.Tensor, training: bool, mask: tf.Tensor) -> tf.Tensor:
        """
        Applies the encoder layer to the input tensor.

        Args:
            x (tf.Tensor): The input tensor.
            training (bool): Whether the model is in training mode.
            mask (tf.Tensor): The mask to be applied in the multi-head attention mechanism.

        Returns:
            tf.Tensor: The output tensor after passing through the encoder layer.
        """
        attn_output, _ = self.multi_head_attention(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2


class Encoder(tf.keras.layers.Layer):
    """
    A custom encoder layer for a transformer-based model.

    Args:
        num_layers (int): Number of encoder layers.
        d_model (int): Dimensionality of the model's hidden representations.
        num_heads (int): Number of self-attention heads.
        dff (int): Dimensionality of the feedforward sub-layer.
        input_vocab_size (int): Size of the input vocabulary.
        maximum_position_encoding (int): Maximum position for positional encoding.
        dropout_rate (float): Dropout rate to apply within the encoder.
        cnn_filter_size (int): Size of the CNN filter for local pattern capturing.
        cnn_num_filters (int): Number of filters in the CNN layer.

    Attributes:
        d_model (int): Dimensionality of the model's hidden representations.
        num_layers (int): Number of encoder layers.
        embedding (tf.keras.layers.Embedding): Embedding layer to convert input tokens to dense vectors.
        cnn (tf.keras.layers.Conv1D): 1D Convolutional layer for capturing local patterns.
        pos_encoding: Positional encoding tensor.
        enc_layers: List of EncoderLayer instances.
        dropout: Dropout layer.
        cnn_filter_size (int): The size of the 1D convolutional filter for local pattern capturing.
        cnn_num_filters (int): The number of filters in the 1D convolutional layer.

    Methods:
        call(x, training, mask): Forward pass of the encoder.

    Returns:
        tf.Tensor: Encoder's output tensor.
    """
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
        """
        Forward pass of the encoder.

        Args:
            x (tf.Tensor): Input tensor of shape (batch_size, input_seq_length).
            training (bool): Whether the model is in training mode.
            mask (tf.Tensor): Mask to apply to the self-attention layer.

        Returns:
            tf.Tensor: Encoder's output tensor of shape (batch_size, input_seq_length, d_model).
        """
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        
        x = self.cnn(x)
        
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
        return x



class DecoderLayer(tf.keras.layers.Layer):
    """
    A single layer of the decoder in a Transformer model.

    Args:
        d_model (int): The dimensionality of the model.
        num_heads (int): The number of attention heads.
        dff (int): The number of hidden units in the feed-forward network.
        dropout_rate (float): The dropout rate to apply within the layer.

    Attributes:
        multi_head_attention1 (MultiHeadAttention): The first multi-head attention layer.
        multi_head_attention2 (MultiHeadAttention): The second multi-head attention layer.
        ffn (tf.keras.layers.Dense): Point-wise feed-forward network.
        layernorm1 (tf.keras.layers.LayerNormalization): Layer normalization for the first attention output.
        layernorm2 (tf.keras.layers.LayerNormalization): Layer normalization for the second attention output.
        layernorm3 (tf.keras.layers.LayerNormalization): Layer normalization for the feed-forward network output.
        dropout1 (tf.keras.layers.Dropout): Dropout layer for the first attention output.
        dropout2 (tf.keras.layers.Dropout): Dropout layer for the second attention output.
        dropout3 (tf.keras.layers.Dropout): Dropout layer for the feed-forward network output.

    Returns:
        tf.Tensor: The output tensor of the decoder layer.
    """
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
        """
        Forward pass of the decoder layer.

        Args:
            x (tf.Tensor): The input tensor.
            enc_output (tf.Tensor): The output of the encoder stack.
            training (bool): Whether the model is in training mode.
            look_ahead_mask (tf.Tensor): The mask to apply for self-attention in the decoder.
            padding_mask (tf.Tensor): The mask to apply for padding.

        Returns:
            tf.Tensor: The output tensor of the decoder layer.
        """
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
    """
    Decoder layer in a transformer model.

    Args:
        num_layers (int): Number of decoder layers.
        d_model (int): Dimension of model.
        num_heads (int): Number of attention heads.
        dff (int): Dimension of feedforward network.
        target_vocab_size (int): Size of target vocabulary.
        maximum_position_encoding (int): Maximum position for positional encoding.
        dropout_rate (float): Dropout rate.

    Attributes:
        d_model (int): Dimension of model.
        num_layers (int): Number of decoder layers.
        embedding (tf.keras.layers.Embedding): Embedding layer.
        pos_encoding (tf.Tensor): Positional encoding tensor.
        dec_layers (list): List of decoder layers.
        dropout (tf.keras.layers.Dropout): Dropout layer.

    Methods:
        call(x, enc_output, training, look_ahead_mask, padding_mask) -> tf.Tensor:
            Forward pass through the decoder layer.

    """
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
        """
        Forward pass through the decoder layer.

        Args:
            x (tf.Tensor): Input tensor.
            enc_output (tf.Tensor): Encoder output tensor.
            training (bool): Whether the model is in training mode.
            look_ahead_mask (tf.Tensor): Mask for look-ahead attention.
            padding_mask (tf.Tensor): Mask for padding.

        Returns:
            tf.Tensor: Output tensor.

        """
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
    """
    A custom implementation of the Transformer model for sequence-to-sequence tasks.

    Args:
        num_layers (int): The number of layers in both the encoder and decoder.
        d_model (int): The dimensionality of the model.
        num_heads (int): The number of attention heads in multi-head attention layers.
        dff (int): The dimensionality of the feedforward neural network in the model.
        input_vocab_size (int): The size of the input vocabulary.
        target_vocab_size (int): The size of the target vocabulary.
        pe_input (int): The maximum positional encoding value for the input sequence.
        pe_target (int): The maximum positional encoding value for the target sequence.
        dropout_rate (float): The dropout rate to be applied in the model.
        cnn_filter_size (int): The size of the 1D convolutional filter for local pattern capturing.
        cnn_num_filters (int): The number of filters in the 1D convolutional layer.

    Returns:
        tf.Tensor: The output tensor from the Transformer model.

    """
    def __init__(self, num_layers: int, d_model: int, num_heads: int, dff: int, input_vocab_size: int,
                 target_vocab_size: int, pe_input: int, pe_target: int, dropout_rate: float,
                 cnn_filter_size: int, cnn_num_filters: int):
        super(Transformer, self).__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size,
                               pe_input, dropout_rate, cnn_filter_size, cnn_num_filters)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size,
                               pe_target, dropout_rate)
        self.final_layer = Dense(target_vocab_size)

    def call(self, inp: tf.Tensor, tar: tf.Tensor, training: bool, enc_padding_mask: tf.Tensor,
             look_ahead_mask: tf.Tensor, dec_padding_mask: tf.Tensor) -> tf.Tensor:
        """
        Execute a forward pass through the Transformer model.

        Args:
            inp (tf.Tensor): The input tensor.
            tar (tf.Tensor): The target tensor for teacher forcing in the decoder.
            training (bool): Indicates whether the model is in training mode.
            enc_padding_mask (tf.Tensor): Padding mask for the encoder input.
            look_ahead_mask (tf.Tensor): Look-ahead mask for the decoder input.
            dec_padding_mask (tf.Tensor): Padding mask for the decoder input.

        Returns:
            tf.Tensor: The output tensor from the Transformer model.

        """
        enc_output = self.encoder(inp, training, enc_padding_mask)
        dec_output = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask)
        final_output = self.final_layer(dec_output)
        return final_output
