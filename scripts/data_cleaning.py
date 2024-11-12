import seaborn as sns
import pandas as pd

# Load the 'dricks' dataset with seaborn -> returns a pandas dataframe
dricks = sns.load_dataset('tips')

# Quick look at the first few rows of the dataset to know what is in there. 
print(dricks.head())


# Check for missing values
print("Missing values:\n", dricks.isnull().sum())

# Handle missing values
df_cleaned = dricks.dropna()

# Save the cleaned dataset
df_cleaned.to_csv('data/processed/cleaned_data.csv', index=False)