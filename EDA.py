import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("titanic.csv")

# -----------------------------
# 1. Basic Information
# -----------------------------
print("First 5 Rows")
print(df.head())

print("\nLast 5 Rows")
print(df.tail())

print("\nShape of Dataset")
print(df.shape)

print("\nColumn Names")
print(df.columns)

print("\nDataset Information")
print(df.info())

print("\nStatistical Summary")
print(df.describe())

# -----------------------------
# 2. Missing Values
# -----------------------------
print("\nMissing Values")
print(df.isnull().sum())

plt.figure(figsize=(10,5))
sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
plt.title("Missing Values Heatmap")
plt.show()

# -----------------------------
# 3. Duplicate Values
# -----------------------------
print("\nDuplicate Rows:", df.duplicated().sum())

# -----------------------------
# 4. Survival Analysis
# -----------------------------
sns.countplot(x="Survived", data=df)
plt.title("Survival Count")
plt.show()

print(df["Survived"].value_counts())

# -----------------------------
# 5. Gender Analysis
# -----------------------------
sns.countplot(x="Sex", data=df)
plt.title("Gender Distribution")
plt.show()

# Survival based on Gender
sns.countplot(x="Sex", hue="Survived", data=df)
plt.title("Survival by Gender")
plt.show()

# -----------------------------
# 6. Passenger Class Analysis
# -----------------------------
sns.countplot(x="Pclass", data=df)
plt.title("Passenger Class Distribution")
plt.show()

sns.countplot(x="Pclass", hue="Survived", data=df)
plt.title("Survival by Passenger Class")
plt.show()

# -----------------------------
# 7. Age Distribution
# -----------------------------
plt.figure(figsize=(8,5))
sns.histplot(df["Age"], bins=30, kde=True)
plt.title("Age Distribution")
plt.show()

# -----------------------------
# 8. Fare Distribution
# -----------------------------
plt.figure(figsize=(8,5))
sns.histplot(df["Fare"], bins=30, kde=True)
plt.title("Fare Distribution")
plt.show()

# -----------------------------
# 9. Embarked Analysis
# -----------------------------
sns.countplot(x="Embarked", data=df)
plt.title("Passengers by Embarked Port")
plt.show()

# -----------------------------
# 10. Outlier Detection
# -----------------------------
plt.figure(figsize=(8,5))
sns.boxplot(x=df["Age"])
plt.title("Age Outliers")
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(x=df["Fare"])
plt.title("Fare Outliers")
plt.show()

# -----------------------------
# 11. Correlation Analysis
# -----------------------------
numeric_df = df.select_dtypes(include=["int64", "float64"])

plt.figure(figsize=(10,6))
sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm"
)
plt.title("Correlation Heatmap")
plt.show()

# -----------------------------
# 12. Pair Plot
# -----------------------------
sns.pairplot(
    df[["Survived", "Pclass", "Age", "Fare"]]
)
plt.show()