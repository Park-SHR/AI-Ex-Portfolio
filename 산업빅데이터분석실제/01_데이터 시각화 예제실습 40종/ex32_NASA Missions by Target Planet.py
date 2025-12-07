import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# NASA Mission Launch sample data
# -----------------------------
data = {
    "Mission": [
        "Mariner 2", "Voyager 1", "Voyager 2", "Pioneer 10", "Pioneer 11",
        "Galileo", "Cassini", "New Horizons", "Juno", "Perseverance"
    ],
    "Launch_Year": [1962, 1977, 1977, 1972, 1973, 1989, 1997, 2006, 2011, 2020],
    "Target": [
        "Venus", "Jupiter", "Neptune", "Jupiter", "Saturn",
        "Jupiter", "Saturn", "Pluto", "Jupiter", "Mars"
    ]
}

df = pd.DataFrame(data)


plt.figure(figsize=(6,6))
df["Target"].value_counts().plot(kind="pie", autopct="%1.1f%%", startangle=90)
plt.title("NASA Missions by Target Planet")
plt.ylabel("")
plt.tight_layout()
plt.show()
