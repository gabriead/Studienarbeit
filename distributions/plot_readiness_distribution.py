import pandas as pd
import matplotlib.pyplot as plt
path = "complete_dataset"
df = pd.read_csv(path)
df = df.drop("Unnamed: 0", axis=1)
df = df.drop("date", axis=1)


# CAUTION:dates are currently not being used for debugging reasons
columnNames = ["daily_load", "fatigue", "mood", "readiness", "sleep_duration", "sleep_quality", "soreness", "stress"]

# filter teams

df_team_a = df[df["player_name_x"].str.startswith("TeamA")]
df_team_b = df[df["player_name_x"].str.startswith("TeamB")]

plt.hist(df_team_a['readiness'], alpha=0.45, color='green')
plt.hist(df_team_b['readiness'], alpha=0.45, color='blue')

plt.legend(['Distribution of readiness in Team A',
            'Distribution of readiness in Team B'])

#plt.show()
plt.savefig("readiness_distribution.png")
