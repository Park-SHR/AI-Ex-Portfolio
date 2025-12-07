import pandas as pd
import matplotlib.pyplot as plt


import pandas as pd
import matplotlib.pyplot as plt
url = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports-extended.dat"
cols = ["AirportID","Name","City","Country","IATA","ICAO","Latitude","Longitude","Altitude","Timezone","DST","Tz","Type","Source"]
df = pd.read_csv(url, header=None, names=cols)
plt.figure()
plt.hist(df['Altitude'].dropna(), bins=50)
plt.xlabel("Altitude (ft)")
plt.ylabel("Count")
plt.title("OpenFlights Airports: Altitude distribution")
plt.tight_layout()
plt.show()
