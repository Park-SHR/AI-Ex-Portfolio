import pandas as pd
import matplotlib.pyplot as plt


import pandas as pd
import matplotlib.pyplot as plt
url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/significant_month.csv"
df = pd.read_csv(url)
d = df.nlargest(10, 'mag')
plt.figure()
plt.bar(d['place'], d['mag'])
plt.xticks(rotation=90)
plt.ylabel("Magnitude")
plt.title("USGS Significant Earthquakes (last month)")
plt.tight_layout()
plt.show()
