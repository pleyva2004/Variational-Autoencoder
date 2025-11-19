import mlx.core as mx
import numpy as np
from sklearn.datasets import fetch_openml
import pickle

def load_mnist():
    """Load MNIST dataset"""
    print("Loading MNIST...")
    X, y = fetch_openml('mnist_784', version=1, parser='auto', return_X_y=True)

    X = X.to_numpy().astype(np.float32) / 255.0  # Normalize to [0, 1]
    y = y.to_numpy().astype(np.int32)

    # Train/test split (standard MNIST split)
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

    # Convert to MLX arrays
    X_train = mx.array(X_train)
    X_test = mx.array(X_test)
    y_train = mx.array(y_train)
    y_test = mx.array(y_test)

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def get_batch(X, batch_size, shuffle=True):
    """Simple batch generator"""
    n = X.shape[0]
    indices = np.arange(n)
    if shuffle:
        np.random.shuffle(indices)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_indices = indices[start:end]
        yield X[mx.array(batch_indices)]

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_mnist()

    # Test batch generation
    for batch in get_batch(X_train, batch_size=128):
        print(f"Batch shape: {batch.shape}")
        break
