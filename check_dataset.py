import pandas as pd

# Load the dataset
data = pd.read_csv("dataset/fake_job_postings.csv")

# Display basic information
print("Dataset Shape:", data.shape)

print("\nColumn Names:")
print(data.columns.tolist())

print("\nFirst 5 Rows:")
print(data.head())

print("\nFake vs Real Jobs:")
print(data["fraudulent"].value_counts())