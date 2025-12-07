from sklearn.datasets import load_linnerud
import pandas as pd
import matplotlib.pyplot as plt


lin = load_linnerud(as_frame=True)
df = lin.frame
plt.figure()
plt.scatter(df['Weight'], df['Chins'])
plt.xlabel("Weight")
plt.ylabel("Chins")
plt.title("Linnerud: Weight vs Chins")
plt.tight_layout()
plt.show()
