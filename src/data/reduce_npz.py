import numpy as np

def sample_npz(input_path, output_path, n_samples, seed=42):
    """
    Randomly sample n_samples from NPZ file while preserving order within sample
    """
    np.random.seed(seed)
    
    # Load with mmap to avoid memory issues
    data = np.load(input_path, mmap_mode='r')
    arr = list(data.values())[0]  # Get first array
    total = len(arr)
    
    # Random sample of indices
    indices = np.random.choice(total, size=min(n_samples, total), replace=False)
    indices.sort()  # Sort to maintain some locality
    
    # Load only the sampled rows
    sampled = arr[indices]
    
    # Save compressed
    np.savez_compressed(output_path, sampled)
    print(f"Saved {len(sampled)} samples to {output_path}")
    return sampled

# Usage
sample_npz('data/raw/train_combined.npz', 'data/processed/train_small.npz', n_samples=50000)
sample_npz('data/raw/target_combined.npz', 'data/processed/target_small.npz', n_samples=50000)