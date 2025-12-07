import pandas as pd
import matplotlib.pyplot as plt

url = "https://raw.githubusercontent.com/fivethirtyeight/data/master/us-weather-history/KCLT.csv" # Trying Charlotte, NC data as an example
try:
    df_weather = pd.read_csv(url)
    plt.figure()
    plt.plot(df_weather["date"], df_weather["actual_mean_temp"])
    plt.title("US Weather History (Charlotte) Temp")
    plt.xlabel("Date")
    plt.ylabel("Mean Temperature")
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"Error fetching or processing data: {e}")