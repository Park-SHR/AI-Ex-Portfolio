from sklearn.datasets import load_digits
import numpy as np
import matplotlib.pyplot as plt


digits = load_digits()
X = digits.data
y = digits.target
import numpy as np
avg = [X[y==k].mean() for k in range(10)]
plt.figure()
plt.bar(range(10), avg)
plt.xlabel("Digit class")
plt.ylabel("Average pixel intensity")
plt.title("Digits dataset: avg intensity per class")
plt.tight_layout()
plt.show()
