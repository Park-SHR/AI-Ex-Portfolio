import pandas as pd
import matplotlib.pyplot as plt


import pandas as pd
url = "https://github.com/owid/co2-data/raw/master/owid-co2-data.csv"
df = pd.read_csv(url)
plt.figure()
plt.plot(df["year"], df["co2"])

plt.title("OWID CO2 Emissions (World)")
plt.tight_layout()
plt.show()
