"""Flax NNX MLP model and training utilities for data parallel tutorial."""

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
import optax
from typing import Tuple, List
import time


class MLP(nnx.Module):
    """A simple multi-layer perceptron using Flax NNX.
    
    Attributes:
        layers: List of linear layers
    """
    
    def __init__(self, layer_sizes: List[int], *, rngs: nnx.Rngs):
        """Initialize the MLP.
        
        Args:
            layer_sizes: List of layer dimensions [input_dim, hidden1, ..., output_dim]
            rngs: Random number generators
        """
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            layer = nnx.Linear(layer_sizes[i], layer_sizes[i + 1], rngs=rngs)
            self.layers.append(layer)
    
    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass through the MLP.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        activations = x
        
        # Apply each layer with ReLU activation except the last layer
        for i, layer in enumerate(self.layers[:-1]):
            activations = layer(activations)
            activations = jax.nn.relu(activations)
        
        # Last layer without activation (for regression)
        return self.layers[-1](activations)


def mse_loss(model: MLP, x_batch: jax.Array, y_batch: jax.Array) -> jax.Array:
    """Mean squared error loss function.
    
    Args:
        model: MLP model
        x_batch: Input batch
        y_batch: Target batch
        
    Returns:
        Scalar loss value
    """
    predictions = model(x_batch)
    return jnp.mean((predictions - y_batch) ** 2)


def train_step(model: MLP, 
               optimizer: nnx.Optimizer,
               x_batch: jax.Array, 
               y_batch: jax.Array) -> jax.Array:
    """Perform a single training step.
    
    Args:
        model: MLP model
        optimizer: Optimizer instance
        x_batch: Input batch
        y_batch: Target batch
        
    Returns:
        Loss value
    """
    loss_value, grads = nnx.value_and_grad(lambda m: mse_loss(m, x_batch, y_batch))(model)
    optimizer.update(grads)
    return loss_value


def generate_synthetic_data(num_samples: int,
                          input_dim: int,
                          seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data from a quadratic function with noise.
    
    Args:
        num_samples: Number of samples to generate
        input_dim: Input dimension
        seed: Random seed
        
    Returns:
        Tuple of (inputs, targets)
    """
    np.random.seed(seed)
    
    # Generate random input features
    x = np.random.randn(num_samples, input_dim)
    
    # Generate target values: sum of squares with noise
    y_clean = np.sum(x ** 2, axis=1, keepdims=True)
    noise = 0.1 * np.random.randn(num_samples, 1)
    y = y_clean + noise
    
    return x.astype(np.float32), y.astype(np.float32)


def create_data_loader(x_data: np.ndarray,
                      y_data: np.ndarray,
                      batch_size: int,
                      num_epochs: int,
                      shuffle: bool = True):
    """Create a simple data loader generator.
    
    Args:
        x_data: Input data
        y_data: Target data
        batch_size: Batch size
        num_epochs: Number of epochs
        shuffle: Whether to shuffle data
        
    Yields:
        Batches of (x_batch, y_batch)
    """
    num_samples = len(x_data)
    steps_per_epoch = num_samples // batch_size
    
    for epoch in range(num_epochs):
        if shuffle:
            perm = np.random.permutation(num_samples)
            x_shuffled = x_data[perm]
            y_shuffled = y_data[perm]
        else:
            x_shuffled = x_data
            y_shuffled = y_data
            
        for step in range(steps_per_epoch):
            idx_start = step * batch_size
            idx_end = idx_start + batch_size
            yield x_shuffled[idx_start:idx_end], y_shuffled[idx_start:idx_end]


def train_model(model: MLP,
                optimizer: nnx.Optimizer,
                x_data: np.ndarray,
                y_data: np.ndarray,
                batch_size: int,
                num_epochs: int,
                verbose: bool = True) -> Tuple[List[float], float]:
    """Train the MLP model.
    
    Args:
        model: MLP model
        optimizer: Optimizer instance
        x_data: Training inputs
        y_data: Training targets
        batch_size: Batch size
        num_epochs: Number of epochs
        verbose: Whether to print progress
        
    Returns:
        Tuple of (losses, training_time)
    """
    # JIT compile the training step
    train_step_jit = nnx.jit(train_step)
    
    losses = []
    num_samples = len(x_data)
    steps_per_epoch = num_samples // batch_size
    
    start_time = time.time()
    
    data_loader = create_data_loader(x_data, y_data, batch_size, num_epochs)
    
    for epoch in range(num_epochs):
        epoch_losses = []
        
        for _ in range(steps_per_epoch):
            x_batch, y_batch = next(data_loader)
            loss = train_step_jit(model, optimizer, x_batch, y_batch)
            epoch_losses.append(float(loss))
        
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(avg_loss)
        
        if verbose:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
    
    end_time = time.time()
    training_time = end_time - start_time
    
    return losses, training_time