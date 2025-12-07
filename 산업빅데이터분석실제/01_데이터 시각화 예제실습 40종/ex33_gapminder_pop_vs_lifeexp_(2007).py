import pandas as pd
import matplotlib.pyplot as plt

# Set the year to explore
year_to_explore = 2002 # Change this value to explore other years

# Try a different URL for the gapminder data
url = "https://raw.githubusercontent.com/plotly/datasets/master/gapminderDataFiveYear.csv"
df = pd.read_csv(url)

# Filter data for the specified year
y_data = df[df['year']==year_to_explore]

plt.figure()
plt.scatter(y_data['pop'], y_data['lifeExp'], alpha=0.6)
plt.xscale("log")
plt.xlabel("Population (log)")
plt.ylabel("Life expectancy")
plt.title(f"Gapminder {year_to_explore}: Population vs LifeExp") # Update title with the year
plt.tight_layout()
plt.show()