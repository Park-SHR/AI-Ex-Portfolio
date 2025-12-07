import pandas as pd
import matplotlib.pyplot as plt


import pandas as pd
url = "https://github.com/owid/energy-data/raw/master/owid-energy-data.csv"
df = pd.read_csv(url)
plt.figure()
plt.plot(df["year"], df["renewables_share_energy"])

plt.title("Renewable Electricity Share (World)")
plt.tight_layout()
plt.show()
