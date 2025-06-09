"""JAX MLP model and training utilities for data parallel tutorial."""

import jax
import jax.numpy as jnp
from typing import List, Tuple, Any
import time


def init_mlp_params(layer_sizes: List[int], key: jax.Array) -> List[Tuple[jax.Array, jax.Array]]:
    """Initialize parameters for a simple MLP.
    
    Args:
        layer_sizes: List of layer dimensions [input_dim, hidden1, ..., output_dim]
        key: JAX random key
    
    Returns:
        List of (weight, bias) tuples for each layer
    """
    keys = jax.random.split(key, len(layer_sizes))
    params = []
    
    for i in range(len(layer_sizes) - 1):
        in_dim, out_dim = layer_sizes[i], layer_sizes[i + 1]
        w_key, b_key = jax.random.split(keys[i])
        
        # Initialize weights with scaled normal distribution
        w = jax.random.normal(w_key, (in_dim, out_dim)) * jnp.sqrt(2.0 / in_dim)
        b = jnp.zeros((out_dim,))
        
        params.append((w, b))
    
    return params


def mlp_forward(params: List[Tuple[jax.Array, jax.Array]], x: jax.Array) -> jax.Array:
    """Forward pass through the MLP.
    
    Args:
        params: List of (weight, bias) tuples for each layer
        x: Input tensor
    
    Returns:
        Output tensor
    """
    activations = x
    
    # Apply each layer with ReLU activation except the last layer
    for i, (w, b) in enumerate(params[:-1]):
        activations = jnp.dot(activations, w) + b
        activations = jax.nn.relu(activations)
    
    # Last layer without activation (for regression)
    w, b = params[-1]
    output = jnp.dot(activations, w) + b
    
    return output


def mse_loss(params: List[Tuple[jax.Array, jax.Array]], x_batch: jax.Array, y_batch: jax.Array) -> jax.Array:
    """Mean squared error loss function.
    
    Args:
        params: Model parameters
        x_batch: Input batch
        y_batch: Target batch
    
    Returns:
        Scalar loss value
    """
    predictions = mlp_forward(params, x_batch)
    return jnp.mean((predictions - y_batch) ** 2)


def train_step(params: List[Tuple[jax.Array, jax.Array]], 
               x_batch: jax.Array, 
               y_batch: jax.Array, 
               learning_rate: float) -> Tuple[List[Tuple[jax.Array, jax.Array]], jax.Array]:
    """Perform a single training step using gradient descent.
    
    Args:
        params: Model parameters
        x_batch: Input batch
        y_batch: Target batch
        learning_rate: Learning rate for gradient descent
    
    Returns:
        Updated parameters and loss value
    """
    # Compute loss and gradients
    loss_value, grads = jax.value_and_grad(mse_loss)(params, x_batch, y_batch)
    
    # Update parameters using gradient descent
    new_params = [(w - learning_rate * dw, b - learning_rate * db)
                 for (w, b), (dw, db) in zip(params, grads)]
    
    return new_params, loss_value


def generate_synthetic_data(num_samples: int, 
                          input_dim: int, 
                          key: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """Generate synthetic data from a quadratic function with noise.
    
    Args:
        num_samples: Number of samples to generate
        input_dim: Input dimension
        key: JAX random key
    
    Returns:
        Tuple of (inputs, targets)
    """
    x_key, noise_key = jax.random.split(key)
    
    # Generate random input features
    x = jax.random.normal(x_key, (num_samples, input_dim))
    
    # Generate target values: sum of squares with noise
    y_clean = jnp.sum(x ** 2, axis=1, keepdims=True)
    noise = 0.1 * jax.random.normal(noise_key, (num_samples, 1))
    y = y_clean + noise
    
    return x, y


def train_model(params: List[Tuple[jax.Array, jax.Array]],
                x_data: jax.Array,
                y_data: jax.Array,
                learning_rate: float,
                batch_size: int,
                num_epochs: int,
                key: jax.Array,
                verbose: bool = True) -> Tuple[List[Tuple[jax.Array, jax.Array]], List[float], float]:
    """Train the MLP model.
    
    Args:
        params: Initial model parameters
        x_data: Training inputs
        y_data: Training targets
        learning_rate: Learning rate
        batch_size: Batch size
        num_epochs: Number of epochs
        key: JAX random key for data shuffling
        verbose: Whether to print progress
    
    Returns:
        Tuple of (trained_params, losses, training_time)
    """
    # JIT-compile the training step for better performance
    train_step_jit = jax.jit(train_step)
    
    num_samples = len(x_data)
    steps_per_epoch = num_samples // batch_size
    
    losses = []
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Shuffle data at the beginning of each epoch
        key, subkey = jax.random.split(key)
        perm = jax.random.permutation(subkey, num_samples)
        x_shuffled = x_data[perm]
        y_shuffled = y_data[perm]
        
        epoch_losses = []
        
        for step in range(steps_per_epoch):
            # Get batch
            idx_start = step * batch_size
            idx_end = idx_start + batch_size
            x_batch = x_shuffled[idx_start:idx_end]
            y_batch = y_shuffled[idx_start:idx_end]
            
            # Perform training step
            params, loss = train_step_jit(params, x_batch, y_batch, learning_rate)
            epoch_losses.append(loss)
        
        # Compute average loss for this epoch
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(avg_loss)
        
        if verbose:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
    
    end_time = time.time()
    training_time = end_time - start_time
    
    return params, losses, training_time