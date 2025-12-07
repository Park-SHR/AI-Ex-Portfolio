import pandas as pd
import matplotlib.pyplot as plt


import pandas as pd
import matplotlib.pyplot as plt
url = "https://raw.githubusercontent.com/fivethirtyeight/data/master/bechdel/movies.csv"
df = pd.read_csv(url)
df['pass'] = (df['binary']==1).astype(int)
year_pass = df.groupby('year')['pass'].mean().reset_index()
plt.figure()
plt.plot(year_pass['year'], year_pass['pass'])
plt.ylabel("Share passing")
plt.title("Bechdel Test: Share of movies passing by year")
plt.tight_layout()
plt.show()
