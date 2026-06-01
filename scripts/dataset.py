import pandas as pd
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing(data_home="./data", as_frame=True)

df = data.frame
df.to_csv("california_housing.csv", index=False)

print(df.head())