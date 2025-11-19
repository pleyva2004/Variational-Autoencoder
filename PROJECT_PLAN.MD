# VAE Project: Complete Mentorship Plan

**Student:** Pablo Leyva  
**Course:** Statistical Learning - Neural Networks Project  
**Platform:** MLX (Apple Silicon M4 Pro)  
**Timeline:** 4 weeks  

---

## **PROJECT OVERVIEW**

### **Goals**
1. **Understand VAE theory deeply** - Master ELBO, variational inference, reparameterization trick
2. **Implement from scratch** - Build VAE in MLX (no high-level abstractions)
3. **Empirical validation** - Demonstrate VAE learns meaningful representations
4. **Statistical comparison** - Rigorously compare to PCA and k-NN baselines

### **Objectives**
1. Implement working VAE on MNIST in MLX
2. Derive and explain ELBO mathematically
3. Visualize latent space and generation quality
4. Compare anomaly detection: VAE vs PCA vs k-NN
5. Produce publication-quality analysis and report

### **Deliverables**

**Code:**
- `vae.py` - VAE model implementation in MLX
- `train.py` - Training loop with logging
- `experiments.py` - All experiments and visualizations
- `baselines.py` - PCA and k-NN implementations
- `utils.py` - Helper functions
- `data_loader.py` - MNIST data loading utilities

**Analysis:**
- ELBO decomposition plots
- Reconstruction quality comparisons
- Latent space visualizations (2D projections, interpolations)
- Generation samples
- ROC curves for anomaly detection (VAE vs PCA vs k-NN)
- Statistical significance tests

**Report:**
- 10-12 page technical report
- Mathematical ELBO derivation
- Experimental results with interpretation
- Comparison analysis

---

## **WEEK-BY-WEEK PLAN**

---

# **WEEK 1: Theory + Basic Implementation**

## **Day 1 (Monday): Deep Dive into ELBO**

### **Morning: Mathematical Foundations (3-4 hours)**

**Deliverable:** Handwritten derivation of ELBO

#### **Task 1.1: Review Prerequisites**
Review these concepts (30 min each):
- KL divergence definition and properties
- Bayes theorem
- Expectation notation E_q[·]
- Jensen's inequality

#### **Task 1.2: ELBO Derivation (write this out by hand)**

Start with the marginal log-likelihood:
```
Goal: Maximize log p(x) for observed data x

log p(x) = log ∫ p(x,z) dz
         = log ∫ p(x|z)p(z) dz
```

**Problem:** This integral is intractable for complex p(x|z)

**Solution:** Introduce approximate posterior q(z|x)

**Full derivation:**
```
log p(x) = log ∫ p(x,z) dz

Multiply by q(z|x)/q(z|x) inside:
= log ∫ (q(z|x)/q(z|x)) p(x,z) dz
= log ∫ q(z|x) [p(x,z)/q(z|x)] dz

This is expectation over q:
= log E_q[p(x,z)/q(z|x)]

Apply Jensen's inequality (log is concave):
log E[X] ≥ E[log X]

≥ E_q[log(p(x,z)/q(z|x))]
= E_q[log p(x,z) - log q(z|x)]
= E_q[log p(x|z)p(z) - log q(z|x)]
= E_q[log p(x|z)] + E_q[log p(z) - log q(z|x)]
= E_q[log p(x|z)] + E_q[log p(z)/q(z|x)]
= E_q[log p(x|z)] - E_q[log q(z|x)/p(z)]
= E_q[log p(x|z)] - KL(q(z|x) || p(z))

This is the ELBO!
```

**ELBO = E_q[log p(x|z)] - KL(q(z|x) || p(z))**

Where:
- First term: **Reconstruction likelihood** (decoder quality)
- Second term: **KL divergence** (regularization, how close q is to prior p)

#### **Task 1.3: Understand Each Term**

**Reconstruction term: E_q[log p(x|z)]**
- In practice: Sample z ~ q(z|x), decode to get x̂, measure how close to x
- For images with Bernoulli pixels: Binary cross-entropy
- For continuous: MSE (Gaussian likelihood)

**KL term: KL(q(z|x) || p(z))**
- Measures difference between learned q(z|x) and prior p(z)
- Prior: p(z) = N(0, I) (standard normal)
- Approximate posterior: q(z|x) = N(μ(x), σ²(x)) (encoder outputs)

**Closed form KL divergence** between two Gaussians:
```
KL(N(μ, σ²) || N(0, 1)) = 0.5 * Σ(σ² + μ² - 1 - log(σ²))
```

This is what we'll implement!

#### **Task 1.4: Reparameterization Trick**

**Problem:** Can't backpropagate through sampling z ~ N(μ, σ²)

**Solution:** Reparameterize
```
Instead of: z ~ N(μ, σ²)
Write: z = μ + σ * ε, where ε ~ N(0, 1)
```

Now gradient flows through μ and σ, not the random sampling!

**Deliverable Check:** You should have handwritten:
- [ ] Full ELBO derivation
- [ ] Closed-form KL divergence formula
- [ ] Reparameterization trick explanation

### **Afternoon: MLX Setup + Data Loading (2-3 hours)**

#### **Task 1.5: MLX Environment Setup**

```bash
# Install MLX
pip install mlx

# Verify installation
python -c "import mlx.core as mx; print(mx.__version__)"
```

#### **Task 1.6: MNIST Data Loading**

Create `data_loader.py`:

```python
import mlx.core as mx
import numpy as np
from sklearn.datasets import fetch_openml
import pickle

def load_mnist():
    """Load MNIST dataset"""
    print("Loading MNIST...")
    mnist = fetch_openml('mnist_784', version=1, parser='auto')
    
    X = mnist.data.astype(np.float32) / 255.0  # Normalize to [0, 1]
    y = mnist.target.astype(np.int32)
    
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
        yield mx.array(X[batch_indices])

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_mnist()
    
    # Test batch generation
    for batch in get_batch(X_train, batch_size=128):
        print(f"Batch shape: {batch.shape}")
        break
```

**Test it:**
```bash
python data_loader.py
```

**Deliverable Check:**
- [ ] MLX installed and working
- [ ] MNIST loads successfully
- [ ] Batch generation works

---

## **Day 2 (Tuesday): VAE Architecture Implementation**

### **Morning: Encoder & Decoder (3-4 hours)**

Create `vae.py`:

```python
import mlx.core as mx
import mlx.nn as nn

class Encoder(nn.Module):
    """Encoder network: x -> (mu, logvar)"""
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
    
    def __call__(self, x):
        # x shape: (batch_size, 784)
        h = mx.maximum(self.fc1(x), 0)  # ReLU activation
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class Decoder(nn.Module):
    """Decoder network: z -> x_reconstructed"""
    def __init__(self, latent_dim=20, hidden_dim=400, output_dim=784):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def __call__(self, z):
        # z shape: (batch_size, latent_dim)
        h = mx.maximum(self.fc1(z), 0)  # ReLU
        x_recon = mx.sigmoid(self.fc2(h))  # Sigmoid for [0,1] pixel values
        return x_recon

class VAE(nn.Module):
    """Variational Autoencoder"""
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)
        self.latent_dim = latent_dim
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = mu + sigma * epsilon
        where epsilon ~ N(0, 1)
        """
        std = mx.exp(0.5 * logvar)
        eps = mx.random.normal(mu.shape)
        z = mu + std * eps
        return z
    
    def __call__(self, x):
        """
        Forward pass
        Returns: x_recon, mu, logvar
        """
        # Encode
        mu, logvar = self.encoder(x)
        
        # Sample latent code
        z = self.reparameterize(mu, logvar)
        
        # Decode
        x_recon = self.decoder(z)
        
        return x_recon, mu, logvar
    
    def generate(self, num_samples=1):
        """Generate new samples by sampling from prior N(0,I)"""
        z = mx.random.normal((num_samples, self.latent_dim))
        x_gen = self.decoder(z)
        return x_gen

def vae_loss(x, x_recon, mu, logvar):
    """
    VAE loss = Reconstruction loss + KL divergence
    
    Args:
        x: original input (batch_size, 784)
        x_recon: reconstructed input (batch_size, 784)
        mu: latent mean (batch_size, latent_dim)
        logvar: latent log variance (batch_size, latent_dim)
    
    Returns:
        total_loss, recon_loss, kl_loss
    """
    # Reconstruction loss (binary cross-entropy)
    # BCE = -[x*log(x_recon) + (1-x)*log(1-x_recon)]
    eps = 1e-8  # for numerical stability
    recon_loss = -mx.sum(
        x * mx.log(x_recon + eps) + (1 - x) * mx.log(1 - x_recon + eps),
        axis=1
    )
    recon_loss = mx.mean(recon_loss)
    
    # KL divergence: KL(N(mu, sigma^2) || N(0, 1))
    # = 0.5 * sum(sigma^2 + mu^2 - 1 - log(sigma^2))
    # = 0.5 * sum(exp(logvar) + mu^2 - 1 - logvar)
    kl_loss = -0.5 * mx.sum(
        1 + logvar - mu**2 - mx.exp(logvar),
        axis=1
    )
    kl_loss = mx.mean(kl_loss)
    
    total_loss = recon_loss + kl_loss
    
    return total_loss, recon_loss, kl_loss

# Test the model
if __name__ == "__main__":
    # Create model
    model = VAE(input_dim=784, hidden_dim=400, latent_dim=20)
    
    # Test forward pass
    batch_size = 16
    x = mx.random.uniform(shape=(batch_size, 784))
    
    x_recon, mu, logvar = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Reconstruction shape: {x_recon.shape}")
    print(f"Mu shape: {mu.shape}")
    print(f"Logvar shape: {logvar.shape}")
    
    # Test loss
    total_loss, recon_loss, kl_loss = vae_loss(x, x_recon, mu, logvar)
    print(f"\nTotal loss: {total_loss.item():.4f}")
    print(f"Recon loss: {recon_loss.item():.4f}")
    print(f"KL loss: {kl_loss.item():.4f}")
    
    # Test generation
    samples = model.generate(num_samples=8)
    print(f"\nGenerated samples shape: {samples.shape}")
```

**Test it:**
```bash
python vae.py
```

**Deliverable Check:**
- [ ] Encoder outputs mu and logvar
- [ ] Reparameterization works
- [ ] Decoder reconstructs
- [ ] Loss computation works (splits into recon + KL)
- [ ] Generation from prior works

### **Afternoon: Training Loop (3 hours)**

Create `train.py`:

```python
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from vae import VAE, vae_loss
from data_loader import load_mnist, get_batch
import json
from datetime import datetime

def loss_fn(model, x):
    """Compute loss for a batch"""
    x_recon, mu, logvar = model(x)
    total_loss, recon_loss, kl_loss = vae_loss(x, x_recon, mu, logvar)
    return total_loss, (recon_loss, kl_loss)

def train_epoch(model, optimizer, X_train, batch_size):
    """Train for one epoch"""
    total_loss = 0
    total_recon = 0
    total_kl = 0
    num_batches = 0
    
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    
    for batch in get_batch(X_train, batch_size, shuffle=True):
        # Forward pass and compute gradients
        (loss, (recon, kl)), grads = loss_and_grad_fn(model, batch)
        
        # Update parameters
        optimizer.update(model, grads)
        
        # Evaluate the updated model
        mx.eval(model.parameters(), optimizer.state)
        
        total_loss += loss.item()
        total_recon += recon.item()
        total_kl += kl.item()
        num_batches += 1
    
    return {
        'loss': total_loss / num_batches,
        'recon': total_recon / num_batches,
        'kl': total_kl / num_batches
    }

def evaluate(model, X_test, batch_size):
    """Evaluate on test set"""
    total_loss = 0
    total_recon = 0
    total_kl = 0
    num_batches = 0
    
    for batch in get_batch(X_test, batch_size, shuffle=False):
        x_recon, mu, logvar = model(batch)
        loss, recon, kl = vae_loss(batch, x_recon, mu, logvar)
        
        total_loss += loss.item()
        total_recon += recon.item()
        total_kl += kl.item()
        num_batches += 1
    
    return {
        'loss': total_loss / num_batches,
        'recon': total_recon / num_batches,
        'kl': total_kl / num_batches
    }

def train_vae(
    input_dim=784,
    hidden_dim=400,
    latent_dim=20,
    batch_size=128,
    learning_rate=1e-3,
    num_epochs=50
):
    """Main training function"""
    
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_mnist()
    
    print("Initializing model...")
    model = VAE(input_dim, hidden_dim, latent_dim)
    optimizer = optim.Adam(learning_rate=learning_rate)
    
    print(f"\nTraining VAE:")
    print(f"  Latent dim: {latent_dim}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Epochs: {num_epochs}\n")
    
    # Training history
    history = {
        'train_loss': [],
        'train_recon': [],
        'train_kl': [],
        'test_loss': [],
        'test_recon': [],
        'test_kl': []
    }
    
    for epoch in range(num_epochs):
        # Train
        train_metrics = train_epoch(model, optimizer, X_train, batch_size)
        
        # Evaluate
        test_metrics = evaluate(model, X_test, batch_size)
        
        # Log
        history['train_loss'].append(train_metrics['loss'])
        history['train_recon'].append(train_metrics['recon'])
        history['train_kl'].append(train_metrics['kl'])
        history['test_loss'].append(test_metrics['loss'])
        history['test_recon'].append(test_metrics['recon'])
        history['test_kl'].append(test_metrics['kl'])
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
              f"Recon: {train_metrics['recon']:.4f}, KL: {train_metrics['kl']:.4f}")
        print(f"  Test  - Loss: {test_metrics['loss']:.4f}, "
              f"Recon: {test_metrics['recon']:.4f}, KL: {test_metrics['kl']:.4f}")
    
    # Save model and history
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save model weights
    model_path = f"models/vae_model_{timestamp}.npz"
    model.save_weights(model_path)
    print(f"\nModel saved to {model_path}")
    
    # Save training history
    history_path = f"models/history_{timestamp}.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"History saved to {history_path}")
    
    return model, history

if __name__ == "__main__":
    import os
    os.makedirs("models", exist_ok=True)
    
    model, history = train_vae(
        latent_dim=20,
        hidden_dim=400,
        batch_size=128,
        learning_rate=1e-3,
        num_epochs=50
    )
```

**Run training:**
```bash
python train.py
```

**Deliverable Check:**
- [ ] Training loop runs without errors
- [ ] Loss decreases over epochs
- [ ] Both reconstruction and KL losses are tracked
- [ ] Model and history are saved

---

## **Day 3 (Wednesday): Visualization & Analysis Tools**

### **All Day: Build Experiment Framework (6-8 hours)**

Create `experiments.py`:

```python
import mlx.core as mx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from vae import VAE, vae_loss
from data_loader import load_mnist
import json

def plot_training_curves(history_path):
    """Plot training curves from history"""
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Total loss
    axes[0].plot(epochs, history['train_loss'], label='Train', linewidth=2)
    axes[0].plot(epochs, history['test_loss'], label='Test', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Total Loss')
    axes[0].set_title('Total Loss (ELBO)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Reconstruction loss
    axes[1].plot(epochs, history['train_recon'], label='Train', linewidth=2)
    axes[1].plot(epochs, history['test_recon'], label='Test', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Reconstruction Loss')
    axes[1].set_title('Reconstruction Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # KL divergence
    axes[2].plot(epochs, history['train_kl'], label='Train', linewidth=2)
    axes[2].plot(epochs, history['test_kl'], label='Test', linewidth=2)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('KL Divergence')
    axes[2].set_title('KL Divergence')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/training_curves.png', dpi=300, bbox_inches='tight')
    print("Saved: figures/training_curves.png")
    plt.close()

def plot_reconstructions(model, X_test, num_samples=10):
    """Plot original vs reconstructed images"""
    # Get random samples
    indices = np.random.choice(X_test.shape[0], num_samples, replace=False)
    samples = X_test[indices]
    
    # Reconstruct
    x_recon, mu, logvar = model(samples)
    
    # Convert to numpy for plotting
    samples_np = np.array(samples).reshape(-1, 28, 28)
    recon_np = np.array(x_recon).reshape(-1, 28, 28)
    
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples*2, 4))
    
    for i in range(num_samples):
        # Original
        axes[0, i].imshow(samples_np[i], cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', fontsize=12)
        
        # Reconstructed
        axes[1, i].imshow(recon_np[i], cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstructed', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('figures/reconstructions.png', dpi=300, bbox_inches='tight')
    print("Saved: figures/reconstructions.png")
    plt.close()

def plot_generations(model, num_samples=64):
    """Plot generated samples from prior"""
    # Generate samples
    x_gen = model.generate(num_samples=num_samples)
    x_gen_np = np.array(x_gen).reshape(-1, 28, 28)
    
    # Plot in grid
    grid_size = int(np.sqrt(num_samples))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            axes[i, j].imshow(x_gen_np[idx], cmap='gray')
            axes[i, j].axis('off')
    
    plt.suptitle('Generated Samples from Prior N(0, I)', fontsize=16)
    plt.tight_layout()
    plt.savefig('figures/generations.png', dpi=300, bbox_inches='tight')
    print("Saved: figures/generations.png")
    plt.close()

def plot_latent_space_2d(model, X_test, y_test):
    """
    Plot 2D latent space (requires model with latent_dim=2)
    """
    if model.latent_dim != 2:
        print("Warning: Latent space visualization requires latent_dim=2")
        print(f"Current latent_dim={model.latent_dim}. Skipping.")
        return
    
    # Encode all test data
    mu, logvar = model.encoder(X_test)
    z = np.array(mu)
    labels = np.array(y_test)
    
    # Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(z[:, 0], z[:, 1], c=labels, cmap='tab10', alpha=0.6, s=5)
    plt.colorbar(scatter, label='Digit')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.title('2D Latent Space Representation of MNIST')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/latent_space_2d.png', dpi=300, bbox_inches='tight')
    print("Saved: figures/latent_space_2d.png")
    plt.close()

def plot_latent_interpolation(model, X_test, num_interpolations=10):
    """Interpolate between two digits in latent space"""
    # Pick two random samples
    idx1, idx2 = np.random.choice(X_test.shape[0], 2, replace=False)
    x1, x2 = X_test[idx1:idx1+1], X_test[idx2:idx2+1]
    
    # Encode to get latent representations
    mu1, _ = model.encoder(x1)
    mu2, _ = model.encoder(x2)
    
    # Interpolate
    alphas = np.linspace(0, 1, num_interpolations)
    interpolations = []
    
    for alpha in alphas:
        z_interp = (1 - alpha) * mu1 + alpha * mu2
        x_interp = model.decoder(z_interp)
        interpolations.append(np.array(x_interp).reshape(28, 28))
    
    # Plot
    fig, axes = plt.subplots(1, num_interpolations, figsize=(num_interpolations*2, 2))
    
    for i, img in enumerate(interpolations):
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f'α={alphas[i]:.1f}', fontsize=10)
    
    plt.suptitle('Latent Space Interpolation', fontsize=14)
    plt.tight_layout()
    plt.savefig('figures/interpolation.png', dpi=300, bbox_inches='tight')
    print("Saved: figures/interpolation.png")
    plt.close()

def analyze_latent_distributions(model, X_test, latent_dim):
    """Analyze if latent distributions approximate N(0,1)"""
    # Encode test set
    mu, logvar = model.encoder(X_test)
    mu_np = np.array(mu)
    std_np = np.exp(0.5 * np.array(logvar))
    
    # Plot distributions for each latent dimension
    num_dims_to_plot = min(4, latent_dim)
    fig, axes = plt.subplots(2, num_dims_to_plot, figsize=(4*num_dims_to_plot, 8))
    
    from scipy import stats
    x_range = np.linspace(-4, 4, 100)
    standard_normal = stats.norm.pdf(x_range, 0, 1)
    
    for i in range(num_dims_to_plot):
        # Plot mu distribution
        axes[0, i].hist(mu_np[:, i], bins=50, density=True, alpha=0.7, label='Learned μ')
        axes[0, i].plot(x_range, standard_normal, 'r-', linewidth=2, label='N(0,1)')
        axes[0, i].set_xlabel('Value')
        axes[0, i].set_ylabel('Density')
        axes[0, i].set_title(f'Latent Dim {i} - Mean')
        axes[0, i].legend()
        axes[0, i].grid(True, alpha=0.3)
        
        # Plot std distribution
        axes[1, i].hist(std_np[:, i], bins=50, density=True, alpha=0.7, label='Learned σ')
        axes[1, i].axvline(1, color='r', linestyle='--', linewidth=2, label='Target σ=1')
        axes[1, i].set_xlabel('Value')
        axes[1, i].set_ylabel('Density')
        axes[1, i].set_title(f'Latent Dim {i} - Std Dev')
        axes[1, i].legend()
        axes[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/latent_distributions.png', dpi=300, bbox_inches='tight')
    print("Saved: figures/latent_distributions.png")
    plt.close()

def run_all_experiments(model_path, history_path):
    """Run all experiments"""
    print("\n=== Running All Experiments ===\n")
    
    # Load data
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_mnist()
    
    # Load model
    print("Loading model...")
    model = VAE(input_dim=784, hidden_dim=400, latent_dim=20)
    model.load_weights(model_path)
    
    # Create figures directory
    import os
    os.makedirs("figures", exist_ok=True)
    
    # Run experiments
    print("\n1. Plotting training curves...")
    plot_training_curves(history_path)
    
    print("\n2. Plotting reconstructions...")
    plot_reconstructions(model, X_test, num_samples=10)
    
    print("\n3. Plotting generations...")
    plot_generations(model, num_samples=64)
    
    print("\n4. Analyzing latent distributions...")
    analyze_latent_distributions(model, X_test, latent_dim=20)
    
    print("\n5. Plotting latent interpolation...")
    plot_latent_interpolation(model, X_test, num_interpolations=10)
    
    # Note: 2D latent space requires retraining with latent_dim=2
    print("\n6. For 2D latent space visualization, train a model with latent_dim=2")
    
    print("\n=== All experiments complete! ===")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python experiments.py <model_path> <history_path>")
        print("Example: python experiments.py models/vae_model_20241119_143022.npz models/history_20241119_143022.json")
        sys.exit(1)
    
    model_path = sys.argv[1]
    history_path = sys.argv[2]
    
    run_all_experiments(model_path, history_path)
```

**Deliverable Check:**
- [ ] Training curves plotted
- [ ] Reconstructions visualized
- [ ] Generations visualized
- [ ] Latent distributions analyzed
- [ ] Interpolation works

**End of Day 3 Review:** You should now have a working VAE with comprehensive visualizations!

---

## **Day 4 (Thursday): Train 2D Model + Deep ELBO Analysis**

### **Morning: Train 2D Latent VAE (2 hours)**

Modify `train.py` to train with `latent_dim=2`:

```bash
# Edit train.py main section:
model, history = train_vae(
    latent_dim=2,  # Change this!
    hidden_dim=400,
    batch_size=128,
    learning_rate=1e-3,
    num_epochs=50
)
```

Run:
```bash
python train.py
```

Then visualize 2D latent space:
```bash
python experiments.py models/vae_model_<timestamp>.npz models/history_<timestamp>.json
```

This will give you beautiful 2D latent space visualizations!

### **Afternoon: ELBO Analysis Document (4-5 hours)**

Create `docs/elbo_analysis.md` with detailed writeup:

```markdown
# ELBO Analysis: Theoretical Foundations and Empirical Validation

## 1. Introduction

Why do we need ELBO? The core problem in VAEs...

## 2. Mathematical Derivation

[Include your handwritten derivation typed up]

### 2.1 Starting from Maximum Likelihood
### 2.2 Introducing the Variational Distribution
### 2.3 Jensen's Inequality
### 2.4 Final ELBO Form

## 3. Understanding Each Component

### 3.1 Reconstruction Term: E_q[log p(x|z)]
### 3.2 KL Divergence Term: KL(q(z|x) || p(z))
### 3.3 The Tradeoff

## 4. Reparameterization Trick

Why we need it, how it works, gradient flow...

## 5. Empirical Validation

[Include plots from your experiments showing:]
- How ELBO evolves during training
- Decomposition into reconstruction + KL
- Latent distributions approximating N(0,1)

## 6. Conclusion
```

**Deliverable Check:**
- [ ] 2D latent model trained
- [ ] 2D latent space visualization created
- [ ] ELBO analysis document complete with theory + experiments

---

## **Day 5 (Friday): Week 1 Review & Baseline Prep**

### **Morning: Code Review & Testing (2-3 hours)**

Go through all your code:
- [ ] Add comprehensive docstrings everywhere
- [ ] Add type hints
- [ ] Test edge cases
- [ ] Clean up any TODOs
- [ ] Ensure all functions have clear comments

### **Afternoon: Baseline Implementation Prep (2-3 hours)**

Start `baselines.py`:

```python
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt

class PCABaseline:
    """PCA for anomaly detection via reconstruction error"""
    def __init__(self, n_components=20):
        self.pca = PCA(n_components=n_components)
        self.n_components = n_components
    
    def fit(self, X_normal):
        """
        Fit PCA on normal data only
        
        Args:
            X_normal: Normal training data (n_samples, n_features)
        """
        print(f"Fitting PCA with {self.n_components} components...")
        self.pca.fit(X_normal)
        explained_var = np.sum(self.pca.explained_variance_ratio_)
        print(f"Explained variance: {explained_var:.4f}")
    
    def reconstruction_error(self, X):
        """
        Compute reconstruction error for each sample
        
        Args:
            X: Input data (n_samples, n_features)
        
        Returns:
            errors: Reconstruction error for each sample (n_samples,)
        """
        X_compressed = self.pca.transform(X)
        X_reconstructed = self.pca.inverse_transform(X_compressed)
        errors = np.mean((X - X_reconstructed) ** 2, axis=1)
        return errors

class KNNBaseline:
    """k-NN for anomaly detection via distance to nearest neighbors"""
    def __init__(self, n_neighbors=5):
        self.knn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
        self.n_neighbors = n_neighbors
    
    def fit(self, X_normal):
        """
        Fit k-NN on normal data
        
        Args:
            X_normal: Normal training data (n_samples, n_features)
        """
        print(f"Fitting k-NN with {self.n_neighbors} neighbors...")
        self.knn.fit(X_normal)
    
    def anomaly_score(self, X):
        """
        Compute anomaly score as average distance to k nearest neighbors
        
        Args:
            X: Input data (n_samples, n_features)
        
        Returns:
            scores: Average distance to k nearest neighbors (n_samples,)
        """
        distances, _ = self.knn.kneighbors(X)
        scores = np.mean(distances, axis=1)
        return scores

# Placeholder for Week 2 implementation
def compare_anomaly_detection(vae_scores, pca_scores, knn_scores, labels):
    """
    Compare anomaly detection performance across methods
    
    Args:
        vae_scores: Anomaly scores from VAE (reconstruction errors)
        pca_scores: Anomaly scores from PCA
        knn_scores: Anomaly scores from k-NN
        labels: Ground truth labels (0=normal, 1=anomaly)
    
    Returns:
        results: Dictionary with ROC-AUC for each method
    """
    # TODO: Implement in Week 2
    pass

if __name__ == "__main__":
    print("Baseline models ready for Week 2!")
    print("TODO: Complete anomaly detection comparison")
```

**Deliverable Check:**
- [ ] Code is clean and well-documented
- [ ] All files have proper structure
- [ ] Baseline classes defined (implementation next week)

### **Weekend Assignment:**

**Reading:**
1. Original VAE paper: Kingma & Welling (2013) "Auto-Encoding Variational Bayes"
2. Review your ELBO derivation - make sure you understand every step
3. Read about anomaly detection with autoencoders

**Preparation for Week 2:**
Think about anomaly detection strategy:
- How will you split MNIST? (Train on some digits, test on others?)
- What metrics matter? (ROC-AUC, precision-recall, detection at 95% specificity?)
- How will you set thresholds fairly across methods?

---

# **WEEK 2: Baselines & Anomaly Detection**

## **Day 6 (Monday): Complete Baseline Implementations**

### **Morning: Finish baselines.py (3-4 hours)**

Add complete anomaly detection comparison:

```python
def compare_anomaly_detection(vae_scores, pca_scores, knn_scores, labels):
    """
    Compare anomaly detection performance across methods
    
    Returns ROC curves, AUC scores, and statistical tests
    """
    from sklearn.metrics import roc_curve, auc, roc_auc_score
    from scipy import stats
    
    results = {}
    
    # Compute ROC curves
    methods = {
        'VAE': vae_scores,
        'PCA': pca_scores,
        'k-NN': knn_scores
    }
    
    plt.figure(figsize=(10, 8))
    
    for name, scores in methods.items():
        fpr, tpr, thresholds = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        results[name] = {
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'auc': roc_auc
        }
        plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {roc_auc:.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves: Anomaly Detection Comparison', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/roc_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: figures/roc_comparison.png")
    plt.close()
    
    # Statistical significance tests
    print("\n=== Statistical Comparison ===")
    print(f"VAE AUC: {results['VAE']['auc']:.4f}")
    print(f"PCA AUC: {results['PCA']['auc']:.4f}")
    print(f"k-NN AUC: {results['k-NN']['auc']:.4f}")
    
    # DeLong test for comparing AUCs would go here
    # For simplicity, we'll use bootstrap confidence intervals
    
    return results
```

### **Afternoon: Anomaly Detection Setup (3-4 hours)**

Create `anomaly_detection.py`:

```python
import mlx.core as mx
import numpy as np
from vae import VAE, vae_loss
from data_loader import load_mnist
from baselines import PCABaseline, KNNBaseline, compare_anomaly_detection

def setup_anomaly_detection_data(normal_digits=[0, 1, 2, 3, 4], 
                                  anomaly_digits=[5, 6, 7, 8, 9]):
    """
    Split MNIST into normal and anomaly classes
    
    Args:
        normal_digits: List of digit classes to treat as "normal"
        anomaly_digits: List of digit classes to treat as "anomalies"
    
    Returns:
        X_train_normal: Training data (normal only)
        X_test_normal: Test data (normal)
        X_test_anomaly: Test data (anomalies)
        y_test: Labels for test data (0=normal, 1=anomaly)
    """
    print("Loading MNIST...")
    X_train, X_test, y_train, y_test = load_mnist()
    
    # Convert to numpy for easier filtering
    X_train_np = np.array(X_train)
    y_train_np = np.array(y_train)
    X_test_np = np.array(X_test)
    y_test_np = np.array(y_test)
    
    # Filter training data (normal only)
    train_mask = np.isin(y_train_np, normal_digits)
    X_train_normal = X_train_np[train_mask]
    
    # Filter test data
    test_normal_mask = np.isin(y_test_np, normal_digits)
    test_anomaly_mask = np.isin(y_test_np, anomaly_digits)
    
    X_test_normal = X_test_np[test_normal_mask]
    X_test_anomaly = X_test_np[test_anomaly_mask]
    
    # Combine test data and create labels
    X_test_combined = np.vstack([X_test_normal, X_test_anomaly])
    y_test_combined = np.hstack([
        np.zeros(len(X_test_normal)),  # 0 = normal
        np.ones(len(X_test_anomaly))   # 1 = anomaly
    ])
    
    print(f"\nDataset split:")
    print(f"  Normal digits: {normal_digits}")
    print(f"  Anomaly digits: {anomaly_digits}")
    print(f"  Train (normal only): {X_train_normal.shape}")
    print(f"  Test normal: {X_test_normal.shape}")
    print(f"  Test anomaly: {X_test_anomaly.shape}")
    
    return X_train_normal, X_test_combined, y_test_combined

def compute_vae_anomaly_scores(model, X):
    """
    Compute VAE anomaly scores (reconstruction error)
    
    Args:
        model: Trained VAE model
        X: Input data
    
    Returns:
        scores: Anomaly scores (higher = more anomalous)
    """
    X_mx = mx.array(X)
    x_recon, mu, logvar = model(X_mx)
    
    # Reconstruction error per sample
    x_recon_np = np.array(x_recon)
    errors = np.mean((X - x_recon_np) ** 2, axis=1)
    
    return errors

def run_anomaly_detection_experiment():
    """Main anomaly detection experiment"""
    
    print("\n" + "="*60)
    print("ANOMALY DETECTION EXPERIMENT")
    print("="*60)
    
    # Setup data
    X_train_normal, X_test, y_test = setup_anomaly_detection_data(
        normal_digits=[0, 1, 2, 3, 4],
        anomaly_digits=[5, 6, 7, 8, 9]
    )
    
    # Train VAE on normal data
    print("\n--- Training VAE ---")
    from train import train_vae
    vae_model, _ = train_vae(
        input_dim=784,
        hidden_dim=400,
        latent_dim=20,
        batch_size=128,
        learning_rate=1e-3,
        num_epochs=30  # Fewer epochs for anomaly detection
    )
    
    # Compute VAE scores
    print("\n--- Computing VAE anomaly scores ---")
    vae_scores = compute_vae_anomaly_scores(vae_model, X_test)
    
    # Train PCA baseline
    print("\n--- Training PCA baseline ---")
    pca = PCABaseline(n_components=20)
    pca.fit(X_train_normal)
    pca_scores = pca.reconstruction_error(X_test)
    
    # Train k-NN baseline
    print("\n--- Training k-NN baseline ---")
    knn = KNNBaseline(n_neighbors=5)
    knn.fit(X_train_normal)
    knn_scores = knn.anomaly_score(X_test)
    
    # Compare methods
    print("\n--- Comparing methods ---")
    results = compare_anomaly_detection(vae_scores, pca_scores, knn_scores, y_test)
    
    # Save results
    import json
    with open('results/anomaly_detection_results.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        results_serializable = {
            method: {
                'auc': float(data['auc']),
                'fpr': data['fpr'].tolist(),
                'tpr': data['tpr'].tolist()
            }
            for method, data in results.items()
        }
        json.dump(results_serializable, f, indent=2)
    
    print("\n=== Experiment complete! ===")
    print("Results saved to results/anomaly_detection_results.json")
    
    return results

if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    results = run_anomaly_detection_experiment()
```

**Deliverable Check:**
- [ ] Baseline implementations complete
- [ ] Anomaly detection data split working
- [ ] All three methods (VAE, PCA, k-NN) can compute anomaly scores
- [ ] ROC comparison working

---

## **Day 7 (Tuesday): Run Full Anomaly Detection Experiments**

### **All Day: Comprehensive Experiments (6-8 hours)**

Run multiple experiment configurations:

1. **Different normal/anomaly splits:**
   - Normal: [0-4], Anomaly: [5-9]
   - Normal: [0-6], Anomaly: [7-9]
   - Normal: even digits, Anomaly: odd digits

2. **Different hyperparameters:**
   - Latent dimensions: 10, 20, 50
   - PCA components: 10, 20, 50
   - k-NN neighbors: 3, 5, 10

3. **Analysis:**
   - Precision-Recall curves
   - Detection rate at fixed false positive rates
   - Confusion matrices at optimal thresholds
   - Statistical significance tests

Create `run_all_experiments.py`:

```python
# Script to run all experiment configurations
# Track results in a structured way
# Generate comprehensive comparison plots
```

**Deliverable Check:**
- [ ] Multiple experiment configurations run
- [ ] Results tracked systematically
- [ ] Comparison plots generated
- [ ] Statistical tests performed

---

## **Day 8 (Wednesday): Statistical Analysis**

### **Morning: Statistical Tests (3-4 hours)**

Create `statistical_analysis.py`:

```python
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

def bootstrap_auc_ci(scores, labels, n_bootstrap=1000, confidence=0.95):
    """
    Compute bootstrap confidence interval for AUC
    
    Args:
        scores: Anomaly scores
        labels: Ground truth labels
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level
    
    Returns:
        auc_mean, auc_lower, auc_upper
    """
    auc_scores = []
    n_samples = len(labels)
    
    for _ in range(n_bootstrap):
        # Bootstrap sample
        indices = np.random.choice(n_samples, n_samples, replace=True)
        sample_scores = scores[indices]
        sample_labels = labels[indices]
        
        # Compute AUC for this sample
        auc = roc_auc_score(sample_labels, sample_scores)
        auc_scores.append(auc)
    
    auc_scores = np.array(auc_scores)
    auc_mean = np.mean(auc_scores)
    alpha = 1 - confidence
    auc_lower = np.percentile(auc_scores, 100 * alpha / 2)
    auc_upper = np.percentile(auc_scores, 100 * (1 - alpha / 2))
    
    return auc_mean, auc_lower, auc_upper

def compare_auc_statistical_test(scores1, scores2, labels, method='bootstrap'):
    """
    Statistical test for difference between two AUCs
    
    Args:
        scores1: Anomaly scores from method 1
        scores2: Anomaly scores from method 2
        labels: Ground truth labels
        method: 'bootstrap' or 'delong'
    
    Returns:
        p_value: p-value for null hypothesis that AUCs are equal
    """
    # Simplified bootstrap test
    auc1 = roc_auc_score(labels, scores1)
    auc2 = roc_auc_score(labels, scores2)
    
    diff_observed = auc1 - auc2
    
    # Bootstrap null distribution
    n_bootstrap = 1000
    diffs = []
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(len(labels), len(labels), replace=True)
        boot_labels = labels[indices]
        boot_scores1 = scores1[indices]
        boot_scores2 = scores2[indices]
        
        boot_auc1 = roc_auc_score(boot_labels, boot_scores1)
        boot_auc2 = roc_auc_score(boot_labels, boot_scores2)
        diffs.append(boot_auc1 - boot_auc2)
    
    diffs = np.array(diffs)
    
    # Two-sided p-value
    p_value = np.mean(np.abs(diffs) >= np.abs(diff_observed))
    
    return p_value, diff_observed

# More statistical analysis functions...
```

### **Afternoon: Visualization Deep Dive (3-4 hours)**

Create additional analysis plots:
- Precision-Recall curves
- Score distributions (normal vs anomaly)
- Error analysis (what kinds of images does each method miss?)
- Latent space visualization colored by anomaly score

**Deliverable Check:**
- [ ] Bootstrap confidence intervals computed
- [ ] Statistical significance tests run
- [ ] Additional visualizations created
- [ ] Error analysis complete

---

## **Day 9 (Thursday): Paper Reading & Related Work**

### **All Day: Literature Review (6-8 hours)**

Read and summarize key papers:

1. **Original VAE paper:**
   - Kingma & Welling (2013) "Auto-Encoding Variational Bayes"
   - Take detailed notes on theory

2. **VAE applications:**
   - β-VAE for disentanglement
   - VAE for anomaly detection in practice
   - Comparisons to other generative models

3. **Anomaly detection literature:**
   - Classical methods (PCA, k-NN)
   - Deep learning approaches
   - Benchmark comparisons

Create `docs/related_work.md` with summaries and connections to your work.

**Deliverable Check:**
- [ ] 3+ papers read and summarized
- [ ] Related work document created
- [ ] Connections to your project identified

---

## **Day 10 (Friday): Week 2 Review & Report Outline**

### **Morning: Code Cleanup (2-3 hours)**

Final code review:
- [ ] All experiments reproducible
- [ ] README with instructions
- [ ] Requirements file
- [ ] Clean directory structure

### **Afternoon: Report Outline (3-4 hours)**

Create `docs/report_outline.md`:

```markdown
# VAE for Anomaly Detection: Technical Report Outline

## 1. Introduction (1-2 pages)
- Problem motivation
- Anomaly detection importance
- Why VAE?
- Contributions

## 2. Background (2-3 pages)
### 2.1 Variational Inference
### 2.2 Autoencoders
### 2.3 Variational Autoencoders
### 2.4 Anomaly Detection

## 3. Theory: ELBO Derivation (2-3 pages)
[Your detailed mathematical derivation]

## 4. Methods (2-3 pages)
### 4.1 VAE Architecture
### 4.2 Training Procedure
### 4.3 Baseline Methods
### 4.4 Evaluation Metrics
### 4.5 Experimental Setup

## 5. Results (3-4 pages)
### 5.1 Training Dynamics
### 5.2 Latent Space Analysis
### 5.3 Generation Quality
### 5.4 Anomaly Detection Performance
### 5.5 Comparison to Baselines
### 5.6 Statistical Analysis

## 6. Discussion (1-2 pages)
### 6.1 When VAE Excels
### 6.2 Limitations
### 6.3 Computational Considerations

## 7. Conclusion (0.5-1 page)

## 8. References

## Appendices
- A. Hyperparameter Sensitivity
- B. Additional Visualizations
- C. Code Structure
```

Fill in bullet points under each section.

**Weekend Assignment:**
- Start writing Introduction and Background sections
- Organize all figures and results
- Prepare for final push in Weeks 3-4

---

# **WEEK 3: Report Writing & Advanced Analysis**

## **Day 11-12 (Mon-Tue): Write Theory Sections**

Write Introduction, Background, and Theory sections of report.

**Deliverable Check:**
- [ ] Introduction draft complete
- [ ] Background draft complete
- [ ] ELBO derivation section complete with equations

---

## **Day 13-14 (Wed-Thu): Write Methods & Results**

Write Methods and Results sections.

**Deliverable Check:**
- [ ] Methods section complete
- [ ] Results section complete with all figures
- [ ] Statistical analysis integrated

---

## **Day 15 (Friday): Discussion & Conclusion**

Complete Discussion and Conclusion sections.

**Deliverable Check:**
- [ ] Discussion section complete
- [ ] Conclusion complete
- [ ] References formatted

---

# **WEEK 4: Final Polish & Presentation**

## **Day 16-17 (Mon-Tue): Report Revision**

- Edit for clarity
- Check all math notation
- Ensure figures are high quality
- Proofread thoroughly

---

## **Day 18-19 (Wed-Thu): Presentation Prep**

Create presentation slides (if needed for class):
- 15-20 slides
- Focus on key results
- Practice timing

---

## **Day 20 (Friday): Final Submission**

- [ ] Final report submitted
- [ ] Code repository clean and documented
- [ ] All figures and results organized
- [ ] Presentation ready (if applicable)

---

# **MILESTONES & CHECKPOINTS**

## **End of Week 1:**
- [ ] Working VAE implementation in MLX
- [ ] Training successful on MNIST
- [ ] Basic visualizations working
- [ ] ELBO derivation written up

## **End of Week 2:**
- [ ] Baselines implemented
- [ ] Anomaly detection experiments complete
- [ ] Statistical comparisons done
- [ ] All results collected

## **End of Week 3:**
- [ ] Report 80% complete
- [ ] All sections drafted
- [ ] Figures finalized

## **End of Week 4:**
- [ ] Project complete and submitted!

---

# **COMMUNICATION PROTOCOL**

**Daily Check-ins:**
- End of each day: Send brief update on progress
- Share completed deliverables
- Ask questions if stuck

**Weekly Reviews:**
- End of each week: Comprehensive review meeting
- Discuss progress vs. plan
- Adjust timeline if needed

**When to Ask for Help:**
- Stuck for more than 1 hour on implementation
- Unclear about theory
- Need feedback on writing
- Want to discuss results

---

# **RESOURCES**

**Key Papers:**
1. Kingma & Welling (2013) - Auto-Encoding Variational Bayes
2. Doersch (2016) - Tutorial on Variational Autoencoders
3. An & Cho (2015) - Variational Autoencoder based Anomaly Detection

**Documentation:**
- MLX documentation: https://ml-explore.github.io/mlx/
- NumPy documentation
- Matplotlib documentation
- scikit-learn documentation

**LaTeX for Report:**
- Overleaf or local LaTeX installation
- Template: IEEE or standard academic format

---

# **SUCCESS CRITERIA**

By the end of this project, you will have:

1. ✅ Deep understanding of VAE theory (ELBO, variational inference)
2. ✅ Complete implementation from scratch in MLX
3. ✅ Comprehensive experimental evaluation
4. ✅ Rigorous statistical comparison to baselines
5. ✅ Publication-quality technical report
6. ✅ Reproducible code repository

**This will be an excellent showcase project for:**
- Graduate school applications
- Job interviews in ML/AI
- Understanding advanced generative models
- Statistical learning portfolio

---

**Ready to start Day 1? Let's do this! 🚀**
