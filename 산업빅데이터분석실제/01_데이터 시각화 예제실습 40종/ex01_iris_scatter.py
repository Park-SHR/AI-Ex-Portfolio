# Auto-generated example. Dataset: free/public source.
    from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import pandas as pd


iris = load_iris(as_frame=True)
df = iris.frame
plt.figure()
for name, g in df.groupby(df['target'].map(dict(enumerate(iris.target_names)))):
    plt.scatter(g['sepal length (cm)'], g['petal length (cm)'], label=name, alpha=0.7)
plt.xlabel("Sepal length (cm)")
plt.ylabel("Petal length (cm)")
plt.title("Iris: Sepal vs Petal length")
plt.legend()
plt.tight_layout()
plt.show()
