import pandas as pd
import matplotlib.pyplot as plt

url = "https://raw.githubusercontent.com/fivethirtyeight/data/master/avengers/avengers.csv"
df = pd.read_csv(url, encoding='latin-1')

plt.figure()
plt.bar(df["Gender"][:20], df["Death1"][:20])
plt.xticks(rotation=90)

plt.title("Avengers: First Death Count by Gender")
plt.tight_layout()
plt.show()