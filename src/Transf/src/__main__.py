import numpy as np
import pandas as pd
import tensorflow as tf

from .model import Transformer, create_look_ahead_mask, create_padding_mask
from .train import (D_MODEL, DFF, DROPOUT_RATE, INPUT_VOCAB_SIZE,
                    MAXIMUM_POSITION_ENCODING, NUM_HEADS, NUM_LAYERS,
                    TARGET_SEQ_LENGTH, TARGET_VOCAB_SIZE)


def main():
    data = pd.read_csv("your_data.csv")
    sample_input  = data["input_column_name"].to_numpy(dtype=np.int64)
    sample_target = data["target_column_name"].to_numpy(dtype=np.int64)

    enc_padding_mask = create_padding_mask(sample_input)
    look_ahead_mask  = create_look_ahead_mask(TARGET_SEQ_LENGTH)
    dec_padding_mask = create_padding_mask(sample_target)

    look_ahead_mask  = tf.where(look_ahead_mask  == 0, -np.inf, look_ahead_mask )
    dec_padding_mask = tf.where(dec_padding_mask == 0, -np.inf, dec_padding_mask)

    sample_transformer = Transformer(
        NUM_LAYERS, D_MODEL, NUM_HEADS, DFF,
        INPUT_VOCAB_SIZE, TARGET_VOCAB_SIZE,
        pe_input=MAXIMUM_POSITION_ENCODING,
        pe_target=TARGET_SEQ_LENGTH,
        dropout_rate=DROPOUT_RATE,
    )

    sample_output = sample_transformer(sample_input, sample_target, training=False,
                                    enc_padding_mask=enc_padding_mask,
                                    look_ahead_mask=look_ahead_mask,
                                    dec_padding_mask=dec_padding_mask)

    print(sample_output)

if __name__ == "__main__":
    main()
