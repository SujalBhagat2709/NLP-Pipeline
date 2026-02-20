import pandas as pd

df = pd.read_csv("step1_loaded.csv")

print("Checking Missing Values:")
print(df.isnull().sum())

print("\nChecking Duplicates:")
print(df.duplicated().sum())

df = df.drop_duplicates()

df.to_csv("step2_validated.csv", index=False)

print("Data Validation Completed")