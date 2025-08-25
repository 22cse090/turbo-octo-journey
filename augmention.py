
import time 
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

start_time=time.time()

# Load the original UAV dataset
df = pd.read_csv("C:\\Users\\user\\Downloads\\uav_navigation_dataset.csv")

# Drop rows with any missing values
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# Number of augmented samples to create (e.g., 2x the original size)
augmentation_factor = 2
num_augmented = len(df) * augmentation_factor

# Select numeric columns for augmentation
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
non_numeric_cols = [col for col in df.columns if col not in numeric_cols]

# Create augmented data
augmented_rows = []
for _ in range(augmentation_factor):
    df_aug = df[numeric_cols].copy()
    noise = np.random.normal(loc=0, scale=0.03, size=df_aug.shape)  # ~3% noise
    df_aug = df_aug * (1 + noise)
    df_aug = df_aug.clip(lower=0)  # Avoid negative values
    augmented_rows.append(df_aug)

# Combine original and augmented data
df_augmented = pd.concat([df] + augmented_rows, ignore_index=True)

# Ensure non-numeric columns (if any) are repeated as-is
for col in non_numeric_cols:
    repeated_col = pd.concat([df[col]] * (augmentation_factor + 1), ignore_index=True)
    df_augmented[col] = repeated_col

# Save the augmented dataset
augmented_path = "\\Users\\user\\OneDrive\\Desktop\\my python folder\\augmented_uav_dataset.csv"
df_augmented.to_csv(augmented_path, index=False)

df.shape, df_augmented.shape, augmented_path

