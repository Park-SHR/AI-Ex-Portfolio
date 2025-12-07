import pandas as pd
import matplotlib.pyplot as plt


import pandas as pd
url = "https://raw.githubusercontent.com/fivethirtyeight/data/master/fandango/fandango_score_comparison.csv"
df = pd.read_csv(url)
plt.figure()
plt.scatter(df["Fandango_votes"], df["Fandango_Stars"], alpha=0.7)

plt.title("Fandango Ratings: Votes vs Rating")
plt.tight_layout()
plt.show()
