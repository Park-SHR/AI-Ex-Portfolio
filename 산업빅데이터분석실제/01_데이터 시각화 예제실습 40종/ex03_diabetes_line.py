from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
import pandas as pd


d = load_diabetes(as_frame=True)
X = d['data']
y = d['target']
plt.figure()
plt.scatter(X['bmi'], y, alpha=0.6)
plt.xlabel("BMI (standardized)")
plt.ylabel("Disease progression (target)")
plt.title("Diabetes dataset: Target vs BMI")
plt.tight_layout()
plt.show()
