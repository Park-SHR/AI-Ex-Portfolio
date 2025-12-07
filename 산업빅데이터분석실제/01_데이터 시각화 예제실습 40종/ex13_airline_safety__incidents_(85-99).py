import pandas as pd
import matplotlib.pyplot as plt


import pandas as pd
url = "https://raw.githubusercontent.com/fivethirtyeight/data/master/airline-safety/airline-safety.csv"
df = pd.read_csv(url)
plt.figure()
plt.bar(df["airline"][:20], df["incidents_85_99"][:20])
plt.xticks(rotation=90)

plt.title("Airline Safety: Incidents (85-99)")
plt.tight_layout()
plt.show()
