import numpy as np

from .model   import Transformer
from .trainer import TransformerTrainer

# Hyperparameters
NUM_LAYERS                = 6
D_MODEL                   = 512
NUM_HEADS                 = 8
DFF                       = 2048
INPUT_VOCAB_SIZE          = 10000
TARGET_VOCAB_SIZE         = 8000
DROPOUT_RATE              = 0.1
MAXIMUM_POSITION_ENCODING = 10000
TARGET_SEQ_LENGTH         = 50
CNN_FILTER_SIZE           = 22
CNN_NUM_FILTERS           = 512

# Create a dummy dataset for demonstration purposes
def generate_dummy_data(batch_size, sequence_length):
    return (
        np.random.randint(1, INPUT_VOCAB_SIZE, size=(batch_size, sequence_length)),
        np.random.randint(1, TARGET_VOCAB_SIZE, size=(batch_size, sequence_length)),
    )

if __name__ == "__main__":
    transformer_model = Transformer(NUM_LAYERS, D_MODEL, NUM_HEADS, DFF,
                                    INPUT_VOCAB_SIZE, TARGET_VOCAB_SIZE,
                                    pe_input=MAXIMUM_POSITION_ENCODING,
                                    pe_target=TARGET_SEQ_LENGTH,
                                    dropout_rate=DROPOUT_RATE,
                                    cnn_filter_size=CNN_FILTER_SIZE,
                                    cnn_num_filters=CNN_NUM_FILTERS,
                                    )

    BATCH_SIZE    = 32
    LEARNING_RATE = 0.001

    trainer = TransformerTrainer(transformer_model, BATCH_SIZE, LEARNING_RATE)

    NUM_SAMPLES = 1000
    dataset = [generate_dummy_data(BATCH_SIZE, TARGET_SEQ_LENGTH) for _ in range(NUM_SAMPLES)]

    epochs = 10
    trainer.train(dataset, epochs)
