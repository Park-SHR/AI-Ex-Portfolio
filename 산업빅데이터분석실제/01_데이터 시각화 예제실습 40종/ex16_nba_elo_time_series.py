import pandas as pd
import matplotlib.pyplot as plt


import pandas as pd
import matplotlib.pyplot as plt
url = "https://raw.githubusercontent.com/fivethirtyeight/data/master/nba-elo/nbaallelo.csv"
df = pd.read_csv(url)
team = "CLE"
g = df[df['team_id']==team]
plt.figure()
plt.plot(g['date_game'], g['elo_i'])
plt.title(f"NBA Elo over time: {team}")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
