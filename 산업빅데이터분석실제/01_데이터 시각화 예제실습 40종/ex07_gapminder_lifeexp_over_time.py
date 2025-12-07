import pandas as pd
import matplotlib.pyplot as plt


url = "https://raw.githubusercontent.com/plotly/datasets/master/gapminderDataFiveYear.csv"
df = pd.read_csv(url)
plt.figure()
plt.plot(df["year"], df["lifeExp"])

plt.title("Gapminder LifeExp Over Time")
plt.tight_layout()
plt.show()