import h5py, numpy as np

with h5py.File('data/testtrainingdata.h5', 'r') as f:
    n = min(1500, f['labels'].shape[0])
    labels = f['labels'][:n].astype(np.float32)
    # downsample images 16x by block-averaging -> 64x64, cheap and keeps structure
    imgs = f['images'][:n, ::16, ::16].astype(np.float32)

X = imgs.reshape(n, -1)
X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)   # scale-invariant
img_d = 1.0 - X @ X.T                                        # cosine distance
np.fill_diagonal(img_d, np.inf)

def circ_mae_deg(a, b):
    d = np.arctan2(np.sin(a - b), np.cos(a - b))
    return np.degrees(np.abs(d).mean(axis=-1))

nn = img_d.argmin(axis=1)                      # nearest image neighbour of each example
lab_d = circ_mae_deg(labels, labels[nn])
rand_d = circ_mae_deg(labels, labels[np.random.default_rng(0).permutation(n)])

print(f"label MAE, nearest-image neighbour: {lab_d.mean():.1f}°")
print(f"label MAE, random pairing:          {rand_d.mean():.1f}°")
print(f"closest 10% of image pairs:         {lab_d[np.argsort(img_d.min(axis=1))[:n//10]].mean():.1f}°")