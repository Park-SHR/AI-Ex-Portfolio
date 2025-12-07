import pandas as pd
import matplotlib.pyplot as plt


import pandas as pd
import matplotlib.pyplot as plt
url = "https://raw.githubusercontent.com/fivethirtyeight/data/master/births/US_births_2000-2014_SSA.csv"
df = pd.read_csv(url)
dow = df.groupby('day_of_week')['births'].mean()
plt.figure()
plt.bar(dow.index, dow.values)
plt.xlabel("Day of week (1=Mon)")
plt.ylabel("Avg births")
plt.title("US Births by Day of Week (2000-2014)")
plt.tight_layout()
plt.show()
