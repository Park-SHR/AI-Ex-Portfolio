import pandas as pd
import matplotlib.pyplot as plt


import pandas as pd
url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_month.csv"
df = pd.read_csv(url)
plt.figure()
plt.hist(df["mag"].dropna(), bins=30)
plt.xlabel("mag")
plt.ylabel("Count")

plt.title("USGS Earthquakes (Last Month) Magnitude")
plt.tight_layout()
plt.show()
