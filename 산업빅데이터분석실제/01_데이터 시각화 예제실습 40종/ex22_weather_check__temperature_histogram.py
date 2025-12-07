import pandas as pd
import matplotlib.pyplot as plt


import pandas as pd
url = "https://raw.githubusercontent.com/fivethirtyeight/data/master/weather-check/weather-check.csv"
df = pd.read_csv(url)

plt.figure(figsize=(8, 6))
plt.hist(df['Age'].dropna(), bins=len(df['Age'].unique()))
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Distribution of Age')
plt.tight_layout()
plt.show()
