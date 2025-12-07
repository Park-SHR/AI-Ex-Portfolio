import pandas as pd
import matplotlib.pyplot as plt


import pandas as pd
import matplotlib.pyplot as plt
url = "https://github.com/owid/co2-data/raw/master/owid-co2-data.csv"
df = pd.read_csv(url)
kor = df[df['iso_code']=="KOR"]
plt.figure()
plt.plot(kor['year'], kor['co2_per_capita'])
plt.title("Korea CO2 per capita")
plt.tight_layout()
plt.show()
