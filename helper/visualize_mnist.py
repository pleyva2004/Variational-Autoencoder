import mlx.core as mx
import numpy as np
import matplotlib.pyplot as plt
from data_loader import load_mnist

# Load the data
X_train, X_test, y_train, y_test = load_mnist()

# Convert MLX arrays back to numpy for visualization
X_train_np = np.array(X_train)
y_train_np = np.array(y_train)

# Create a figure with subplots
fig, axes = plt.subplots(3, 5, figsize=(12, 8))
fig.suptitle('Sample MNIST Images', fontsize=16)

# Display 15 random images
indices = np.random.choice(len(X_train_np), 15, replace=False)

for idx, ax in enumerate(axes.flat):
    # Get the image and reshape from 784 to 28x28
    image = X_train_np[indices[idx]].reshape(28, 28)
    label = y_train_np[indices[idx]]

    # Display the image
    ax.imshow(image, cmap='gray')
    ax.set_title(f'Label: {label}')
    ax.axis('off')

plt.tight_layout()
plt.show()

print(f"\nDataset info:")
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Image shape (flattened): {X_train.shape[1]}")
print(f"Image shape (original): 28x28")
print(f"Pixel value range: [{X_train_np.min():.2f}, {X_train_np.max():.2f}]")

