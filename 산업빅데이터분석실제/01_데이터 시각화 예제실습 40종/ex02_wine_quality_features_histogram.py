from sklearn.datasets import load_wine
import pandas as pd
import matplotlib.pyplot as plt


wine = load_wine(as_frame=True)
df = wine.frame
col = 'alcohol'
plt.figure()
plt.hist(df[col], bins=20)
plt.xlabel(col)
plt.ylabel("Count")
plt.title("Wine dataset: Alcohol distribution")
plt.tight_layout()
plt.show()
