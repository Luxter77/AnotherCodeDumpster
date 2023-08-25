"""
Transformer Trainer Module

This module provides the `TransformerTrainer` class,
which facilitates the training of a Transformer model
for sequence-to-sequence tasks.
The TransformerTrainer class encapsulates the training process including
loss calculation, gradient computation, and optimization.
It also supports checkpoints and TensorBoard logging.

Example:
    from transformer_trainer import TransformerTrainer
    from transformer_model import TransformerModel

    # Create a Transformer model
    model = TransformerModel(num_layers=4, d_model=128,
                             num_heads=8, d_ff=512,
                             input_vocab_size=10000,
                             target_vocab_size=8000,
                             max_seq_length=50)

    # Create a TransformerTrainer instance
    trainer = TransformerTrainer(transformer_model=model,
                                 batch_size=32, learning_rate=0.001)

    # Load and preprocess your training data as a dataset
    train_dataset = ...  # Your training dataset

    # Train the Transformer model
    trainer.train(dataset=train_dataset, epochs=10)

Attributes:
    - transformer: The Transformer model to be trained.
    - batch_size: Batch size for training.
    - learning_rate: Learning rate for optimization.
    - log_dir: Directory to save TensorBoard logs.
    - checkpoint_dir: Directory to save model checkpoints.
    - loss_object: Loss function for training.
    - optimizer: Optimization algorithm (Adam) for model updates.
    - train_loss: Metric to track the training loss.
    - summary_writer: TensorBoard summary writer for logging.

Methods:
    - loss_function(real, pred): Calculate the loss value.

    - train_step(input_seq, target_seq, enc_padding_mask,
      look_ahead_mask, dec_padding_mask): Perform a single training step.

    - train(dataset, epochs): Train the Transformer model using
      the provided dataset for the specified number of epochs.
"""

import os
import gc
from typing import Iterable, Tuple

import tensorflow as tf
from keras.losses import SparseCategoricalCrossentropy
from keras.metrics import Mean
from keras.optimizers import Adam
from tqdm.auto import tqdm

from .model import create_look_ahead_mask, create_padding_mask


class TransformerTrainer:
    """
    Facilitates the training process of a Transformer model designed
    for sequence-to-sequence tasks.
    Encapsulates essential functionalities such as loss calculation,
    gradient computation, and optimization.
    Supports features like checkpointing and logging with TensorBoard.

    Args:
        transformer_model (tf.keras.Model): The Transformer model to be trained.
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for optimization.
        log_dir (str, optional): Directory to save TensorBoard logs.
            Defaults to "logs/".
        checkpoint_dir (str, optional): Directory to save model checkpoints.
            Defaults to "checkpoints/".

    Attributes:
        transformer (tf.keras.Model): The Transformer model to be trained.
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for optimization.
        log_dir (str): Directory to save TensorBoard logs.
        checkpoint_dir (str): Directory to save model checkpoints.
        loss_object: Loss function for training.
        optimizer: Optimization algorithm (Adam) for model updates.
        train_loss: Metric to track the training loss.
        summary_writer: TensorBoard summary writer for logging.

    Methods:
        loss_function(real, pred): Calculate the loss value.

        train_step(input_seq, target_seq, enc_padding_mask,
                   look_ahead_mask, dec_padding_mask): Perform a single training step.

        train(dataset, epochs): Train the Transformer model using
        the provided dataset for the specified number of epochs.
    """
    def __init__(self, transformer_model: tf.keras.Model, batch_size: int, learning_rate: float,
                 log_dir: str = "logs/", checkpoint_dir: str = "checkpoints/"):
        """
        Initialize the TransformerTrainer.

        Args:
            transformer_model (tf.keras.Model): The Transformer model to be trained.
            batch_size (int): Batch size for training.
            learning_rate (float): Learning rate for optimization.
            log_dir (str): Directory to save TensorBoard logs.
            checkpoint_dir (str): Directory to save model checkpoints.
        """
        self.transformer = transformer_model
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir

        self.loss_object = SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        self.optimizer = Adam(learning_rate=self.learning_rate)
        self.train_loss = Mean()

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.summary_writer = tf.summary.create_file_writer(self.log_dir)

    def loss_function(self, real: tf.Tensor, pred: tf.Tensor) -> tf.Tensor:
        """
        Calculate the loss value.

        Args:
            real (tf.Tensor): The true labels.
            pred (tf.Tensor): The predicted logits.

        Returns:
            tf.Tensor: The calculated loss.
        """
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

    def train_step(self, input_seq: tf.Tensor, target_seq: tf.Tensor,
                   enc_padding_mask: tf.Tensor, look_ahead_mask: tf.Tensor,
                   dec_padding_mask: tf.Tensor) -> tf.Tensor:
        """
        Perform a single training step.

        Args:
            input_seq (tf.Tensor): Input sequence.
            target_seq (tf.Tensor): Target sequence.
            enc_padding_mask (tf.Tensor): Encoder padding mask.
            look_ahead_mask (tf.Tensor): Look-ahead mask for decoder.
            dec_padding_mask (tf.Tensor): Decoder padding mask.

        Returns:
            tf.Tensor: Batch loss.
        """
        with tf.GradientTape() as tape:
            predictions = self.transformer(input_seq, target_seq[:, :-1], training=True,
                                           enc_padding_mask=enc_padding_mask,
                                           look_ahead_mask=look_ahead_mask,
                                           dec_padding_mask=dec_padding_mask)
            loss = self.loss_function(target_seq[:, 1:], predictions)
            batch_loss = tf.reduce_mean(loss)

        variables = self.transformer.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return batch_loss

    def train(self, dataset: Iterable[Tuple[tf.Tensor, tf.Tensor]], epochs: int):
        """
        Train the Transformer model.

        Args:
            dataset (Iterable[Tuple[tf.Tensor, tf.Tensor]]): Iterable containing input-target pairs.
            epochs (int): Number of training epochs.
        """
        for epoch in range(epochs):
            self.train_loss.reset_states()

            progress_bar = tqdm(dataset, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch")

            for batch, (input_seq, target_seq) in enumerate(progress_bar):
                enc_padding_mask = create_padding_mask(input_seq)
                look_ahead_mask  = create_look_ahead_mask(target_seq.shape[1] - 1)
                dec_padding_mask = create_padding_mask(input_seq)

                batch_loss = self.train_step(input_seq, target_seq, enc_padding_mask,
                                             look_ahead_mask, dec_padding_mask)

                self.train_loss(batch_loss)

                progress_bar.set_postfix({"batch_loss": f"{batch_loss:.4f}"})

                if ((batch % 100) == 0):
                    gc.collect()

            with self.summary_writer.as_default():
                tf.summary.scalar("loss", self.train_loss.result(), step=epoch) # pylint: disable=not-callable

            self.transformer.save_weights(
                os.path.join(self.checkpoint_dir, f"ckpt_epoch_{epoch + 1}")
            )

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {self.train_loss.result():.4f}")  # pylint: disable=not-callable
