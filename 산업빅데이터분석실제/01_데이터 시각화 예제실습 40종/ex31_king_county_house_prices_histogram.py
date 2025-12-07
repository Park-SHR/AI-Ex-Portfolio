import pandas as pd
import matplotlib.pyplot as plt

url = "https://raw.githubusercontent.com/fivethirtyeight/data/master/hate-crimes/hate_crimes.csv"
df = pd.read_csv(url)

plt.figure(figsize=(10, 8)) # Adjust figure size for better readability
plt.bar(df["state"], df["hate_crimes_per_100k_splc"])
plt.xticks(rotation=90)

plt.title("Hate Crimes per 100k (SPLC) by State")
plt.xlabel("State")
plt.ylabel("Hate Crimes per 100k (SPLC)")
plt.tight_layout()
plt.show()