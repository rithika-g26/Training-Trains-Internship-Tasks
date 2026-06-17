import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# =====================================
# Dataset
# =====================================
data = {
    "Age": [22, 38, 26, 35, 35, 54, 2, 27, 14, 30],
    "Pclass": [3, 1, 3, 1, 3, 1, 3, 3, 2, 2],
    "Fare": [7.25, 71.28, 7.92, 53.10, 8.05, 51.86, 21.07, 11.13, 30.07, 26.00]
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)

# =====================================
# Standardization
# =====================================
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# =====================================
# PCA
# =====================================
pca = PCA(n_components=2)

principal_components = pca.fit_transform(scaled_data)

# =====================================
# PCA DataFrame
# =====================================
pca_df = pd.DataFrame(
    principal_components,
    columns=["Principal Component 1", "Principal Component 2"]
)

print("\nPCA Transformed Data:")
print(pca_df)

# =====================================
# Explained Variance
# =====================================
print("\nExplained Variance Ratio:")
print(pca.explained_variance_ratio_)

print("\nTotal Variance Explained:")
print(sum(pca.explained_variance_ratio_))