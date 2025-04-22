"""
CXR Foundation: Data-Efficient Classifier Training

This script demonstrates how to train a simple neural network classifier using
precomputed ELIXR embeddings (elixr_img_contrastive) from chest X-ray images.
These embeddings are text-aligned image embeddings from the Q-former output in 
ELIXR (https://arxiv.org/abs/2308.01317).

The script assumes precomputed embeddings and labels from the NIH ChestX-ray14 dataset.
"""

import numpy as np
import tensorflow as tf
import tensorflow_models as tfm
from typing import Dict, Iterable, List, Optional, Tuple, Union
from official.modeling.optimization import lars
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for classifier model."""
    token_num: int = 32
    embeddings_size: int = 128
    learning_rate: float = 0.1
    end_lr_factor: float = 1.0
    dropout: float = 0.0
    decay_steps: int = 1000
    hidden_layer_sizes: List[int] = None
    weight_decay: float = 0.0
    seed: Optional[int] = None


def create_tf_dataset_from_embeddings(
    embeddings: Iterable[np.ndarray],
    labels: Iterable[int],
    embeddings_size: int
) -> tf.data.Dataset:
    """Create a tf.data.Dataset from embeddings and labels.
    
    Args:
        embeddings: Iterable of embedding arrays
        labels: Iterable of label values (0 or 1)
        embeddings_size: Size of each embedding vector
        
    Returns:
        A TensorFlow dataset containing (embedding, label) pairs
        
    Raises:
        AssertionError: If the lengths of embeddings and labels don't match
    """
    # Convert to lists if they aren't already
    embeddings_list = list(embeddings)
    labels_list = list(labels)

    # Validate input
    assert len(embeddings_list) == len(labels_list), \
        "Lengths of embeddings and labels must be equal"

    # Ensure correct data type for embeddings
    embeddings_list = [np.asarray(e, dtype=np.float32) for e in embeddings_list]

    # Create datasets
    ds_embeddings = tf.data.Dataset.from_tensor_slices(embeddings_list)
    ds_labels = tf.data.Dataset.from_tensor_slices(labels_list)

    # Combine into a single dataset
    return tf.data.Dataset.zip((ds_embeddings, ds_labels))


def create_model(
    heads: List[str],
    config: ModelConfig,
    loss_weights: Optional[Dict[str, float]] = None
) -> tf.keras.Model:
    """Creates a classifier model using LARS optimizer with cosine decay.
    
    Args:
        heads: List of output head names (usually diagnosis categories)
        config: ModelConfig instance with model parameters
        loss_weights: Optional dictionary mapping head names to loss weights
        
    Returns:
        A compiled TensorFlow Keras model
    """
    # Default for hidden layers if not provided
    hidden_layer_sizes = config.hidden_layer_sizes or [512, 256]
    
    # Define model architecture
    inputs = tf.keras.Input(shape=(config.token_num * config.embeddings_size,))
    inputs_reshape = tf.keras.layers.Reshape((config.token_num, config.embeddings_size))(inputs)
    inputs_pooled = tf.keras.layers.GlobalAveragePooling1D(data_format='channels_last')(inputs_reshape)
    
    # Build hidden layers
    hidden = inputs_pooled
    for size in hidden_layer_sizes:
        hidden = tf.keras.layers.Dense(
            size,
            activation='relu',
            kernel_initializer=tf.keras.initializers.HeUniform(seed=config.seed),
            kernel_regularizer=tf.keras.regularizers.l2(l2=config.weight_decay),
            bias_regularizer=tf.keras.regularizers.l2(l2=config.weight_decay)
        )(hidden)
        hidden = tf.keras.layers.BatchNormalization()(hidden)
        hidden = tf.keras.layers.Dropout(config.dropout, seed=config.seed)(hidden)
    
    # Output layer
    output = tf.keras.layers.Dense(
        units=len(heads),
        activation='sigmoid',
        kernel_initializer=tf.keras.initializers.HeUniform(seed=config.seed)
    )(hidden)

    # Create separate outputs for each head
    outputs = {}
    for i, head in enumerate(heads):
        outputs[head] = tf.keras.layers.Lambda(
            lambda x, idx=i: x[..., idx:idx + 1], 
            name=head.lower()
        )(output)

    # Create and compile model
    model = tf.keras.Model(inputs, outputs)
    
    # Learning rate schedule
    learning_rate_fn = tf.keras.experimental.CosineDecay(
        tf.cast(config.learning_rate, tf.float32),
        tf.cast(config.decay_steps, tf.float32),
        alpha=tf.cast(config.end_lr_factor, tf.float32)
    )
    
    # Set loss weights or use defaults
    loss_weights = loss_weights or {head: 1.0 for head in heads}
    
    # Compile model with metrics
    model.compile(
        optimizer=tfm.optimization.lars.LARS(learning_rate=learning_rate_fn),
        loss={head: 'binary_crossentropy' for head in heads},
        loss_weights=loss_weights,
        weighted_metrics=[
            tf.keras.metrics.FalsePositives(),
            tf.keras.metrics.FalseNegatives(),
            tf.keras.metrics.TruePositives(),
            tf.keras.metrics.TrueNegatives(),
            tf.keras.metrics.AUC(),
            tf.keras.metrics.AUC(curve='PR', name='auc_pr')
        ]
    )
    
    return model


def train_model(
    df_train: dict,
    df_validate: dict,
    diagnosis: str,
    config: ModelConfig,
    epochs: int = 100,
    batch_size: int = 512,
    val_batch_size: int = 1
) -> tf.keras.Model:
    """Train a model on the provided training data.
    
    Args:
        df_train: Dictionary or DataFrame with training data
        df_validate: Dictionary or DataFrame with validation data
        diagnosis: Name of the diagnosis to predict
        config: ModelConfig instance with model parameters
        epochs: Number of training epochs
        batch_size: Batch size for training
        val_batch_size: Batch size for validation
        
    Returns:
        Trained TensorFlow model
    """
    # Prepare datasets
    training_data = create_tf_dataset_from_embeddings(
        embeddings=df_train["embeddings"].values,
        labels=df_train[diagnosis].values,
        embeddings_size=config.token_num * config.embeddings_size
    )

    validation_data = create_tf_dataset_from_embeddings(
        embeddings=df_validate["embeddings"].values,
        labels=df_validate[diagnosis].values,
        embeddings_size=config.token_num * config.embeddings_size
    )

    # Create model
    model = create_model(
        heads=[diagnosis],
        config=config,
    )

    # Train model
    model.fit(
        x=training_data.batch(batch_size).prefetch(tf.data.AUTOTUNE).cache(),
        validation_data=validation_data.batch(val_batch_size).cache(),
        epochs=epochs,
    )

    return model


def main():
    """Main function to run the training process."""
    # This would typically load data from files or a database
    # For demonstration, this is commented out as we don't have the actual data
    # df_train and df_validate would come from data loading code
    
    # Example usage (commented out as we don't have the actual data)
    """
    # Define constants
    DIAGNOSIS = "Cardiomegaly"  # Example diagnosis
    
    # Configure model
    config = ModelConfig(
        token_num=32,
        embeddings_size=128,
        learning_rate=0.1,
        dropout=0.0,
        decay_steps=1000,
        hidden_layer_sizes=[512, 256],
        weight_decay=0.0,
        seed=42
    )
    
    # Train model
    model = train_model(
        df_train=df_train,
        df_validate=df_validate,
        diagnosis=DIAGNOSIS,
        config=config,
        epochs=100
    )
    
    # Display model summary
    model.summary()
    
    # Save the model if needed
    model.save('cxr_classifier_model')
    """


if __name__ == "__main__":
    main()