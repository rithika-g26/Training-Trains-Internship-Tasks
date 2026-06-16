import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Sample Data
data = {
    "Student": ["A", "B", "C", "D", "E"],
    "Marks": [85, 90, 78, 92, 88]
}

df = pd.DataFrame(data)

# Bar Plot
sns.barplot(x="Student", y="Marks", data=df)

plt.title("Student Marks")
plt.show()