import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# =====================================
# Dataset
# =====================================
data = {
    "Age": [22, 38, 26, 35, 35, 54, 2, 27, 14, 30],
    "Pclass": [3, 1, 3, 1, 3, 1, 3, 3, 2, 2],
    "Sex": [0, 1, 1, 1, 0, 0, 0, 1, 1, 1],  # Male=0, Female=1
    "Survived": [0, 1, 1, 1, 0, 0, 0, 1, 1, 1]
}

df = pd.DataFrame(data)

# =====================================
# Features and Target
# =====================================
X = df[["Age", "Pclass", "Sex"]]
y = df["Survived"]

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
# SVM Model
# =====================================
model = SVC(kernel="linear")

model.fit(X_train, y_train)

# =====================================
# Prediction
# =====================================
y_pred = model.predict(X_test)

# =====================================
# Evaluation
# =====================================
print("Accuracy Score:")
print(accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# =====================================
# Results
# =====================================
result = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})

print("\nPredictions:")
print(result)