import matplotlib.pyplot as plt
import time
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error, accuracy_score

print("Alright, grabbing MNIST... might take a sec")
mnist_data = fetch_openml('mnist_784', version=1, as_frame=False)

X_all = mnist_data.data
y_all = mnist_data.target.astype(int)

X_all = X_all / 255.0

X_tr, X_te, y_tr, y_te = train_test_split( X_all, y_all,test_size=10000,random_state=42,stratify=y_all)
scaler_obj = StandardScaler()
X_tr_scaled = scaler_obj.fit_transform(X_tr)
X_te_scaled = scaler_obj.transform(X_te)

X_tr_scaled = X_tr_scaled[:10000]
y_tr = y_tr[:10000]

X_te_scaled = X_te_scaled[:2000]
y_te = y_te[:2000]

component_list = [20, 50, 100]

pca_errs, ica_errs = [], []
pca_scores, ica_scores = [], []
pca_time_log, ica_time_log = [], []
ica_runs_store = {}

print("Running baseline SVM (no PCA/ICA)...")
base_model = SVC(kernel='linear')
base_model.fit(X_tr_scaled, y_tr)

baseline_pred = base_model.predict(X_te_scaled)
base_acc = accuracy_score(y_te, baseline_pred)

print("Baseline acc:", base_acc)

for comp in component_list:
    print("\n--- Trying components =", comp, "---")

    t0 = time.time()

    pca_model = PCA(n_components=comp)
    X_tr_pca = pca_model.fit_transform(X_tr_scaled)
    X_te_pca = pca_model.transform(X_te_scaled)

    pca_time_log.append(time.time() - t0)

    recon_pca = pca_model.inverse_transform(X_te_pca)
    err_pca = mean_squared_error(X_te_scaled, recon_pca)
    pca_errs.append(err_pca)

    svm_pca = SVC(kernel='linear')
    svm_pca.fit(X_tr_pca, y_tr)

    pred_pca = svm_pca.predict(X_te_pca)
    score_pca = accuracy_score(y_te, pred_pca)
    pca_scores.append(score_pca)

    print("PCA -> mse:", round(err_pca, 4), "| acc:", round(score_pca, 4))

    ica_err_list = []
    ica_score_list = []

    t1 = time.time()

    for seed_val in range(3):
        ica_model = FastICA(n_components=comp,random_state=seed_val,max_iter=300)
        X_tr_ica = ica_model.fit_transform(X_tr_scaled)
        X_te_ica = ica_model.transform(X_te_scaled)

        recon_ica = ica_model.inverse_transform(X_te_ica)
        err_ica = mean_squared_error(X_te_scaled, recon_ica)

        svm_ica = SVC(kernel='linear')
        svm_ica.fit(X_tr_ica, y_tr)

        pred_ica = svm_ica.predict(X_te_ica)
        score_ica = accuracy_score(y_te, pred_ica)

        ica_err_list.append(err_ica)
        ica_score_list.append(score_ica)

    ica_time_log.append(time.time() - t1)

    avg_ica_err = np.mean(ica_err_list)
    avg_ica_score = np.mean(ica_score_list)

    ica_errs.append(avg_ica_err)
    ica_scores.append(avg_ica_score)

    ica_runs_store[comp] = ica_score_list

    print("ICA -> mse:", round(avg_ica_err, 4), "| acc:", round(avg_ica_score, 4))

def show_recons(original, pca_img, ica_img, n_samples=5):
    plt.figure(figsize=(10, 6))

    for idx in range(n_samples):
        plt.subplot(n_samples, 3, idx*3 + 1)
        plt.imshow(original[idx].reshape(28, 28), cmap='gray')
        plt.title("Orig")
        plt.axis('off')

        plt.subplot(n_samples, 3, idx*3 + 2)
        plt.imshow(pca_img[idx].reshape(28, 28), cmap='gray')
        plt.title("PCA")
        plt.axis('off')

        plt.subplot(n_samples, 3, idx*3 + 3)
        plt.imshow(ica_img[idx].reshape(28, 28), cmap='gray')
        plt.title("ICA")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

show_recons(X_te_scaled, recon_pca, recon_ica)

plt.figure()
plt.plot(component_list, pca_scores, marker='o', label='PCA')
plt.plot(component_list, ica_scores, marker='o', label='ICA')
plt.axhline(base_acc, linestyle='--', label='baseline')
plt.xlabel("num components")
plt.ylabel("accuracy")
plt.legend()
plt.title("Accuracy comparison")
plt.show()

plt.figure()
plt.plot(component_list, pca_errs, marker='o', label='PCA')
plt.plot(component_list, ica_errs, marker='o', label='ICA')
plt.xlabel("num components")
plt.ylabel("MSE")
plt.legend()
plt.title("Reconstruction error")
plt.show()

pca_full_model = PCA().fit(X_tr_scaled)
cum_var = np.cumsum(pca_full_model.explained_variance_ratio_)

plt.figure()
plt.plot(cum_var)
plt.xlabel("components")
plt.ylabel("cum variance")
plt.title("PCA variance")
plt.show()

ica_var_list = []
for comp in component_list:
    ica_var_list.append(np.var(ica_runs_store[comp]))

plt.figure()
plt.plot(component_list, ica_var_list, marker='o')
plt.xlabel("components")
plt.ylabel("variance in acc")
plt.title("ICA stability")
plt.show()

plt.figure()
plt.plot(component_list, pca_time_log, marker='o', label='PCA')
plt.plot(component_list, ica_time_log, marker='o', label='ICA')
plt.xlabel("components")
plt.ylabel("time (s)")
plt.legend()
plt.title("Time taken")
plt.show()

def show_components(comp_arr, title_txt, n_show=10):
    plt.figure(figsize=(10, 4))

    for i in range(n_show):
        plt.subplot(2, n_show // 2, i + 1)
        plt.imshow(comp_arr[i].reshape(28, 28), cmap='gray')
        plt.axis('off')

    plt.suptitle(title_txt)
    plt.show()

show_components(pca_model.components_, "PCA components")
show_components(ica_model.components_, "ICA components")
