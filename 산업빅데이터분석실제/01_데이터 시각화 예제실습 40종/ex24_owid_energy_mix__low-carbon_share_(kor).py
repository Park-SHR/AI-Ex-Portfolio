import pandas as pd
import matplotlib.pyplot as plt


import pandas as pd
url = "https://github.com/owid/energy-data/raw/master/owid-energy-data.csv"
df = pd.read_csv(url)

# Filter for a specific country, e.g., 'World'
df_world = df[df['country'] == 'World'].copy()

plt.figure()
plt.plot(df_world["year"], df_world["low_carbon_elec_per_capita"])

plt.title("OWID Energy Mix: Low-Carbon Electricity per Capita (World)")
plt.xlabel("Year")
plt.ylabel("Low-Carbon Electricity per Capita")
plt.tight_layout()
plt.show()