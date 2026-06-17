import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# =====================================
# Customer Dataset
# =====================================
data = {
    "CustomerID": [1,2,3,4,5,6,7,8,9,10],
    "Age": [19,21,20,23,31,45,52,23,40,60],
    "Annual_Income": [15,16,17,18,40,60,80,45,70,90],
    "Spending_Score": [39,81,6,77,40,76,6,94,3,10]
}

df = pd.DataFrame(data)

print("Customer Dataset:")
print(df)

# =====================================
# Feature Selection
# =====================================
X = df[["Annual_Income", "Spending_Score"]]

# =====================================
# Feature Scaling
# =====================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =====================================
# K-Means Clustering
# =====================================
kmeans = KMeans(n_clusters=3, random_state=42)

df["Cluster"] = kmeans.fit_predict(X_scaled)

# =====================================
# Results
# =====================================
print("\nCustomer Segments:")
print(df)

# =====================================
# Visualization
# =====================================
plt.scatter(
    df["Annual_Income"],
    df["Spending_Score"],
    c=df["Cluster"]
)

plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation")

plt.show()