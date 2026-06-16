import pandas as pd

# Create a DataFrame
data = {
    "Name": ["Rithika", "Anu", "Kavi"],
    "Age": [20, 21, 22],
    "City": ["Chennai", "Bangalore", "Hyderabad"]
}

df = pd.DataFrame(data)

# Display DataFrame
print("DataFrame:")
print(df)

# Display first rows
print("\nFirst 2 Rows:")
print(df.head(2))

# Display column names
print("\nColumns:")
print(df.columns)

# Display Data Types
print("\nData Types:")
print(df.dtypes)

# Display Statistics
print("\nStatistics:")
print(df.describe())

# Select a Column
print("\nNames:")
print(df["Name"])

# Filter Data
print("\nAge greater than 20:")
print(df[df["Age"] > 20])

# Add New Column
df["Salary"] = [25000, 30000, 35000]
print("\nAfter Adding Salary Column:")
print(df)

# Save to CSV
df.to_csv("students.csv", index=False)
print("\nData saved to students.csv")