import pandas as pd

def get_points(x: str): 
	return float(x[1:-1].split(",")[0])

def get_reason(x: str):
	return x[1:-1].split(",")[1].strip()[1:-1]

def get_timesteps(x: str):
	return int(x[1:-1].split(",")[2])

dir = "outputs/group5"



data = (
    pd.read_csv(f"{dir}/results.csv")
    .melt("Unnamed: 0")
    .rename(columns={"Unnamed: 0": "seed", "variable": "level"})
)
data["points"] = data["value"].apply(get_points)
data["timesteps"] = data["value"].apply(get_timesteps)
data["reason"] = data["value"].apply(get_reason)
del data["value"]

data = data[["level", "seed", "points", "timesteps", "reason"]]

data.to_csv(f"{dir}/results_formatted.csv", index=False)
data.groupby("level", as_index=False)["points"].sum().to_csv(
    f"{dir}/results_by_level.csv", index=False
)

print(f"Total points: {data['points'].sum():.2f}/{len(data)}")