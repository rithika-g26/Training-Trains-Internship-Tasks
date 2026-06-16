import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
# 5. Define Features and Target
# =====================================
X = df.drop("Survived", axis=1)
y = df["Survived"]

# =====================================
# 6. Train-Test Split
# =====================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# =====================================
# 7. Build Linear Regression Model
# =====================================
model = LinearRegression()

model.fit(X_train, y_train)

# =====================================
# 8. Prediction
# =====================================
y_pred = model.predict(X_test)

# =====================================
# 9. Evaluation
# =====================================
print("R2 Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

# =====================================
# 10. Sample Predictions
# =====================================
result = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})

print("\nPredictions:")
print(result.head(10))