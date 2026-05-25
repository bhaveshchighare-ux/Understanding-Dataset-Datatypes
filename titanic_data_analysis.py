from google.colab import files

uploaded = files.upload()
import pandas as pd
import numpy as np
df = pd.read_csv('train.csv')
print("First 5 Rows of Dataset")
df.head()
print("Shape of Dataset:")
print(df.shape)
print("Dataset Information:")
df.info()
print("Statistical Summary:")
df.describe()
print("Missing Values:")
df.isnull().sum()
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns

print("Numerical Columns:")
print(numerical_columns)
categorical_columns = df.select_dtypes(include=['object']).columns

print("Categorical Columns:")
print(categorical_columns)
print("Data Types:")
print(df.dtypes)
