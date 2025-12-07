import pandas as pd
import matplotlib.pyplot as plt


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(url, delimiter=';')
plt.figure()
plt.hist(df["quality"].dropna(), bins=30)
plt.xlabel("quality")
plt.ylabel("Count")

plt.title("UCI Wine Quality (Red): Quality Histogram")
plt.tight_layout()
plt.show()