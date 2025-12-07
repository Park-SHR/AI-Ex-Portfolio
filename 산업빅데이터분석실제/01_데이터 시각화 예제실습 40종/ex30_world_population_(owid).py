import pandas as pd
import matplotlib.pyplot as plt

url = "https://raw.githubusercontent.com/fivethirtyeight/data/master/hate-crimes/hate_crimes.csv"
df = pd.read_csv(url)

plt.figure(figsize=(10, 8)) # Adjust figure size for better readability
plt.bar(df["state"], df["median_household_income"])
plt.xticks(rotation=90)

plt.title("Median Household Income by State (Hate Crimes Dataset)")
plt.xlabel("State")
plt.ylabel("Median Household Income")
plt.tight_layout()
plt.show()