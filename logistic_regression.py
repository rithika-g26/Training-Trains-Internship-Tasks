import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# =====================================
# 1. Load Dataset
# =====================================
df = pd.read_csv("titanic.csv")

# =====================================
# 2. Handle Missing Values
# =====================================
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df["Fare"] = df["Fare"].fillna(df["Fare"].median())

# =====================================
# 3. Drop Unnecessary Columns
# =====================================
columns_to_drop = ["PassengerId", "Name", "Ticket", "Cabin"]

for col in columns_to_drop:
    if col in df.columns:
        df.drop(col, axis=1, inplace=True)

# =====================================
# 4. Encode Categorical Variables
# =====================================
le = LabelEncoder()

df["Sex"] = le.fit_transform(df["Sex"])
df["Embarked"] = le.fit_transform(df["Embarked"])

# =====================================
# 5. Split Features and Target
# =====================================
X = df.drop("Survived", axis=1)
y = df["Survived"]

# =====================================
# 6. Train-Test Split
# =====================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =====================================
# 7. Train Logistic Regression Model
# =====================================
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# =====================================
# 8. Predictions
# =====================================
y_pred = model.predict(X_test)

# =====================================
# 9. Evaluation
# =====================================
print("Accuracy Score:")
print(accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# =====================================
# 10. Sample Predictions
# =====================================
results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})

print("\nSample Predictions:")
print(results.head(10))