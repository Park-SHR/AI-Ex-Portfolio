import pandas as pd
import matplotlib.pyplot as plt


import pandas as pd
import matplotlib.pyplot as plt
url = "https://raw.githubusercontent.com/plotly/datasets/master/gapminderDataFiveYear.csv"
df = pd.read_csv(url)
asia = df[df['continent']=="Asia"]
plt.figure()
plt.scatter(asia['gdpPercap'], asia['lifeExp'], alpha=0.6)
plt.xscale("log")
plt.xlabel("GDP per capita (log)")
plt.ylabel("Life expectancy")
plt.title("Gapminder Asia: Life expectancy vs GDP per capita")
plt.tight_layout()
plt.show()