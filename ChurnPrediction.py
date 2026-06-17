import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# =====================================
# Customer Churn Dataset
# =====================================
data = {
    "Age": [25, 35, 45, 20, 30, 50, 40, 60, 28, 33],
    "MonthlyCharges": [50, 70, 90, 40, 60, 100, 80, 120, 55, 65],
    "Tenure": [1, 5, 10, 2, 4, 12, 8, 15, 3, 6],
    "Churn": [1, 0, 0, 1, 0, 0, 0, 0, 1, 0]
}

df = pd.DataFrame(data)

print("Customer Dataset:")
print(df)

# =====================================
# Features and Target
# =====================================
X = df[["Age", "MonthlyCharges", "Tenure"]]
y = df["Churn"]

# =====================================
# Train-Test Split
# =====================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =====================================
# Logistic Regression Model
# =====================================
model = LogisticRegression()

model.fit(X_train, y_train)

# =====================================
# Prediction
# =====================================
y_pred = model.predict(X_test)

# =====================================
# Evaluation
# =====================================
print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# =====================================
# Sample Predictions
# =====================================
result = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})

print("\nPredictions:")
print(result)