import pandas as pd
import matplotlib.pyplot as plt


import pandas as pd
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
column_names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin', 'car name']
df = pd.read_csv(url, delim_whitespace=True, names=column_names)

plt.figure(figsize=(8, 6))
plt.scatter(df['horsepower'], df['mpg'], alpha=0.7)
plt.xlabel('Horsepower')
plt.ylabel('MPG')
plt.title('MPG vs Horsepower (Auto-MPG Dataset)')
plt.grid(True)
plt.tight_layout()
plt.show()