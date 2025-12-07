import pandas as pd
import matplotlib.pyplot as plt


import pandas as pd
url = "https://raw.githubusercontent.com/fivethirtyeight/data/master/alcohol-consumption/drinks.csv"
df = pd.read_csv(url)
plt.figure()
plt.bar(df["country"][:20], df["total_litres_of_pure_alcohol"][:20])
plt.xticks(rotation=90)

plt.title("Alcohol Consumption per Capita (Top 20)")
plt.tight_layout()
plt.show()
