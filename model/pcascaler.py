import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Misalkan Anda memiliki data X
# Contoh data
X = np.random.rand(3762, 13)  # Data dengan 100 sampel dan 10 fitur

# Skalakan data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Terapkan PCA
pca = PCA(n_components=10)  # Ambil 5 komponen utama
X_pca = pca.fit_transform(X_scaled)

# Simpan objek Skalar dan PCA ke file .pkl
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(pca, 'pca.pkl')
