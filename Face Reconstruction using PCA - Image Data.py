import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA

faces_data = fetch_olivetti_faces()
X = faces_data.data

num_samples, num_features = X.shape

max_components = 200
pca_model = PCA(n_components=max_components)

X_transformed = pca_model.fit_transform(X)

def reconstruct_data(k_components):
    reduced = X_transformed[:, :k_components]
    comps = pca_model.components_[:k_components]
    avg_face = pca_model.mean_
    approx = reduced @ comps + avg_face
    return approx

component_list = [10, 25, 50, 100, 200]

np.random.seed(0)
random_indices = np.random.choice(num_samples, 5, replace=False)

fig, axes = plt.subplots(len(random_indices), len(component_list) + 1, figsize=(12, 8))

for r, sample_idx in enumerate(random_indices):
    axes[r, 0].imshow(X[sample_idx].reshape(64, 64), cmap='gray')
    axes[r, 0].set_title("orig")
    axes[r, 0].axis('off')

    for c, k_val in enumerate(component_list):
        recon_all = reconstruct_data(k_val)
        img = recon_all[sample_idx]

        axes[r, c + 1].imshow(img.reshape(64, 64), cmap='gray')
        axes[r, c + 1].set_title(str(k_val))
        axes[r, c + 1].axis('off')

plt.tight_layout()
plt.show()

mse_errors = []

for k_val in component_list:
    recon_all = reconstruct_data(k_val)
    diff = (X - recon_all) ** 2
    mse = diff.mean()
    mse_errors.append(mse)

cum_explained = np.cumsum(pca_model.explained_variance_ratio_)

fig, ax = plt.subplots(1, 2, figsize=(14, 5))

ax[0].plot(component_list, mse_errors, marker='o')
ax[0].set_xlabel("components")
ax[0].set_ylabel("mse")
ax[0].set_title("reconstruction error")

ax[1].plot(range(1, max_components + 1), cum_explained)
ax[1].set_xlabel("components")
ax[1].set_ylabel("variance")
ax[1].set_title("explained variance")

plt.tight_layout()
plt.show()
