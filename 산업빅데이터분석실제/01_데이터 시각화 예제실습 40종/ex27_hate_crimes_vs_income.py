import pandas as pd
import matplotlib.pyplot as plt

url = "https://raw.githubusercontent.com/fivethirtyeight/data/master/hate-crimes/hate_crimes.csv"
df = pd.read_csv(url)
plt.figure()
plt.scatter(df["median_household_income"], df["avg_hatecrimes_per_100k_fbi"], alpha=0.7)

plt.title("Hate Crimes vs Income")
plt.tight_layout()
plt.show()