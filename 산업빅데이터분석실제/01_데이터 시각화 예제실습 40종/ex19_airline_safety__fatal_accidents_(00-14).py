import pandas as pd
import matplotlib.pyplot as plt


import pandas as pd
url = "https://raw.githubusercontent.com/fivethirtyeight/data/master/airline-safety/airline-safety.csv"
df = pd.read_csv(url)
plt.figure()
plt.bar(df["airline"][:20], df["fatal_accidents_00_14"][:20])
plt.xticks(rotation=90)

plt.title("Airline Safety: Fatal Accidents (00-14)")
plt.tight_layout()
plt.show()
