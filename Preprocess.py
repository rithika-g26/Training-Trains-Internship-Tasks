
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load Dataset
df = pd.read_csv("titanic.csv")

# ----------------------------------
# 1. Check Missing Values
# ----------------------------------
print("Missing Values Before:")
print(df.isnull().sum())

# ----------------------------------
# 2. Handle Missing Values
# ----------------------------------

# Fill Age with Median
df["Age"] = df["Age"].fillna(df["Age"].median())

# Fill Embarked with Mode
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Fill Fare with Median (if any missing)
df["Fare"] = df["Fare"].fillna(df["Fare"].median())

# Drop Cabin (too many missing values)
df.drop("Cabin", axis=1, inplace=True)

# ----------------------------------
# 3. Remove Unnecessary Columns
# ----------------------------------
df.drop(["PassengerId", "Name", "Ticket"], axis=1, inplace=True)

# ----------------------------------
# 4. Encode Categorical Columns
# ----------------------------------
le = LabelEncoder()

df["Sex"] = le.fit_transform(df["Sex"])
df["Embarked"] = le.fit_transform(df["Embarked"])

# ----------------------------------
# 5. Check Duplicates
# ----------------------------------
print("Duplicate Rows:", df.duplicated().sum())

df.drop_duplicates(inplace=True)

# ----------------------------------
# 6. Feature Scaling
# ----------------------------------
scaler = StandardScaler()

df[["Age", "Fare"]] = scaler.fit_transform(
    df[["Age", "Fare"]]
)

# ----------------------------------
# 7. Final Dataset Check
# ----------------------------------
print("\nMissing Values After:")
print(df.isnull().sum())

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

# ----------------------------------
# 8. Save Preprocessed Dataset
# ----------------------------------
df.to_csv("titanic_preprocessed.csv", index=False)

print("\nPreprocessed dataset saved successfully!")