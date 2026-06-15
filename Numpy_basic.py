import numpy as np


arr1 = np.array([10, 20, 30, 40, 50])
arr2 = np.array([1, 2, 3, 4, 5])

print("Array 1:", arr1)
print("Array 2:", arr2)


print("\nAddition:", arr1 + arr2)
print("Subtraction:", arr1 - arr2)
print("Multiplication:", arr1 * arr2)
print("Division:", arr1 / arr2)


print("\nShape:", arr1.shape)
print("Size:", arr1.size)
print("Data Type:", arr1.dtype)

print("\nSum:", np.sum(arr1))
print("Mean:", np.mean(arr1))
print("Maximum:", np.max(arr1))
print("Minimum:", np.min(arr1))


matrix = np.array([[1, 2, 3], [4, 5, 6]])
print("\n2D Array:")
print(matrix)


zeros = np.zeros((2, 3))
ones = np.ones((2, 3))

print("\nZeros Array:")
print(zeros)

print("\nOnes Array:")
print(ones)

# Random Numbers
random_array = np.random.randint(1, 100, size=(3, 3))
print("\nRandom Array:")
print(random_array)