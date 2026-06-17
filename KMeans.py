import pandas as pd
from sklearn.cluster import KMeans

# =====================================
# Dataset
# =====================================
data = {
    "Age": [22, 38, 26, 35, 35, 54, 2, 27, 14, 30],
    "Pclass": [3, 1, 3, 1, 3, 1, 3, 3, 2, 2],
    "Fare": [7.25, 71.28, 7.92, 53.10, 8.05, 51.86, 21.07, 11.13, 30.07, 26.00]
}

df = pd.DataFrame(data)

# =====================================
# Features
# =====================================
X = df[["Age", "Pclass", "Fare"]]

# =====================================
# K-Means Clustering
# =====================================
kmeans = KMeans(n_clusters=2, random_state=42)

df["Cluster"] = kmeans.fit_predict(X)

# =====================================
# Output
# =====================================
print("Cluster Centers:")
print(kmeans.cluster_centers_)

print("\nDataset with Clusters:")
print(df)