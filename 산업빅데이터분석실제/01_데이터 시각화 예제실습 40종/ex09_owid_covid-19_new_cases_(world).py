import pandas as pd
import matplotlib.pyplot as plt


import pandas as pd
url = "https://catalog.ourworldindata.org/garden/covid/latest/compact/compact.csv"
df = pd.read_csv(url)
plt.figure()
plt.plot(df["date"], df["new_cases"])

plt.title("OWID COVID-19 New Cases (World)")
plt.tight_layout()
plt.show()
