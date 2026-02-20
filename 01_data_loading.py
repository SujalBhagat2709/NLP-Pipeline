import pandas as pd

df = pd.read_csv("dataset.csv")

print("Dataset Loaded")
print("Shape:", df.shape)
print(df.head())

df.to_csv("step1_loaded.csv", index=False)