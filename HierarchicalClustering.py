import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

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
# Dendrogram
# =====================================
linked = linkage(df, method='ward')

plt.figure(figsize=(10, 5))
dendrogram(linked)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Data Points")
plt.ylabel("Euclidean Distance")
plt.show()

# =====================================
# Hierarchical Clustering Model
# =====================================
hc = AgglomerativeClustering(
    n_clusters=2,
    metric='euclidean',
    linkage='ward'
)

clusters = hc.fit_predict(df)

# =====================================
# Add Cluster Labels
# =====================================
df["Cluster"] = clusters

print("\nDataset with Cluster Labels:")
print(df)

# =====================================
# Cluster Summary
# =====================================
print("\nCluster Counts:")
print(df["Cluster"].value_counts())