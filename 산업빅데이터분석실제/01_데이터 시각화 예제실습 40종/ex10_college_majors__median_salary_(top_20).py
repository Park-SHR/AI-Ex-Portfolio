import pandas as pd
import matplotlib.pyplot as plt


import pandas as pd
url = "https://raw.githubusercontent.com/fivethirtyeight/data/master/college-majors/recent-grads.csv"
df = pd.read_csv(url)
plt.figure()
plt.bar(df["Major"][:20], df["Median"][:20])
plt.xticks(rotation=90)

plt.title("College Majors: Median Salary (Top 20)")
plt.tight_layout()
plt.show()
