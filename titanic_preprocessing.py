import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split

# =====================================
# 1. Load Dataset
# =====================================
df = pd.read_csv("titanic.csv")

print("Original Dataset Shape:")
print(df.shape)

# =====================================
# 2. Handle Missing Values
# =====================================

# Fill Age with Median
df["Age"] = df["Age"].fillna(df["Age"].median())

# Fill Embarked with Mode
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Fill Fare with Median
df["Fare"] = df["Fare"].fillna(df["Fare"].median())

# Drop Cabin column
df.drop("Cabin", axis=1, inplace=True)

# =====================================
# 3. Drop Unnecessary Columns
# =====================================
df.drop(["PassengerId", "Name", "Ticket"], axis=1, inplace=True)

# =====================================
# 4. Encode Categorical Features
# =====================================
le = LabelEncoder()

df["Sex"] = le.fit_transform(df["Sex"])
df["Embarked"] = le.fit_transform(df["Embarked"])

# =====================================
# 5. Separate Features and Target
# =====================================
X = df.drop("Survived", axis=1)
y = df["Survived"]

print("\nFeatures:")
print(X.head())

# =====================================
# 6. Normalization (Min-Max Scaling)
# =====================================
normalizer = MinMaxScaler()

X_normalized = pd.DataFrame(
    normalizer.fit_transform(X),
    columns=X.columns
)

print("\nNormalized Data:")
print(X_normalized.head())

# =====================================
# 7. Standardization (Z-Score Scaling)
# =====================================
standardizer = StandardScaler()

X_standardized = pd.DataFrame(
    standardizer.fit_transform(X),
    columns=X.columns
)

print("\nStandardized Data:")
print(X_standardized.head())

# =====================================
# 8. Feature Selection
# =====================================
selector = SelectKBest(score_func=chi2, k=4)

X_selected = selector.fit_transform(X_normalized, y)

selected_features = X.columns[selector.get_support()]

print("\nSelected Features:")
print(selected_features)

# =====================================
# 9. Train-Test Split
# =====================================
X_train, X_test, y_train, y_test = train_test_split(
    X_selected,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)

# =====================================
# 10. Final Output
# =====================================
print("\nPreprocessing Completed Successfully!")