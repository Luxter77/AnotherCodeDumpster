import os
from typing import Iterable, Tuple

import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers import Adam
from tqdm.auto import tqdm

from .model import create_padding_mask, create_look_ahead_mask

class TransformerTrainer:
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

                batch_loss = self.train_step(input_seq, target_seq, enc_padding_mask, look_ahead_mask, dec_padding_mask)

                self.train_loss(batch_loss)

                progress_bar.set_postfix({"batch_loss": f"{batch_loss:.4f}"})

            with self.summary_writer.as_default():
                tf.summary.scalar("loss", self.train_loss.result(), step=epoch)

            self.transformer.save_weights(os.path.join(self.checkpoint_dir, f"ckpt_epoch_{epoch + 1}"))

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {self.train_loss.result():.4f}")
