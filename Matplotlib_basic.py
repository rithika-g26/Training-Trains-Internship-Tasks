import matplotlib.pyplot as plt

# Data
subjects = ["Python", "Java", "C++", "SQL"]
marks = [85, 78, 90, 88]

# Create Bar Chart
plt.bar(subjects, marks)

# Title and Labels
plt.title("Student Marks")
plt.xlabel("Subjects")
plt.ylabel("Marks")

# Display Chart
plt.show()