import pandas as pd
import matplotlib.pyplot as plt


import pandas as pd
url = "https://raw.githubusercontent.com/plotly/datasets/master/gapminderDataFiveYear.csv"
df = pd.read_csv(url)
plt.figure(figsize=(8, 6))
asia = df[df['continent'] == 'Asia'].copy()
plt.scatter(asia['gdpPercap'], asia['lifeExp'], alpha=0.6)
plt.xscale("log")
plt.xlabel("GDP per capita (log scale)")
plt.ylabel("Life expectancy")
plt.title("Life Expectancy vs. GDP per Capita in Asia")
plt.grid(True)
plt.tight_layout()
plt.show()