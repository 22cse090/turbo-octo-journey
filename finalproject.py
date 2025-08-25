import time
start_time = time.time()
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load your actual UAV dataset
df = pd.read_csv("C:\\Users\\user\\OneDrive\\Desktop\\my python folder\\augmented_uav_dataset.csv")

# Drop missing values
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

print(" Data cleaned. Shape after preprocessing:", df.shape)

# Feature Engineering
df['lat_long'] = df['latitude'] * df['longitude']

# Battery threshold filtering
BATTERY_THRESHOLD = 40
active_df = df[(df['battery_level'] >= BATTERY_THRESHOLD) & (df['battery_level'] <= 100)]

# Selected features
selected_features = [
    'altitude', 'latitude', 'longitude',
    'wind_speed', 'imu_acc_x', 'imu_gyro_z',
    'lat_long',
]

# Scale features
scaler = StandardScaler()
active_df.loc[:, selected_features] = scaler.fit_transform(active_df[selected_features])


print(f" Filtered active drones with battery >= {BATTERY_THRESHOLD}%.")
print(f" Active data shape: {active_df.shape}")

# Federated Training Simulation
print("\n Simulating training on 15 clients...\n")

client_mses = []
client_r2s = []

for client_id in range(15):
    client_data = active_df.sample(frac=1/15, random_state=client_id)
    X_train = client_data[selected_features]
    y_train = client_data['battery_level']
   
    model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=10,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=client_id
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_train)

    mse = mean_squared_error(y_train, y_pred)
    r2 = r2_score(y_train, y_pred)

    client_mses.append(mse)
    client_r2s.append(r2)

    print(f" Client {client_id+1}: MSE = {mse:.2f}, R² = {r2 * 100:.2f}%")

# Global Metrics
avg_mse = sum(client_mses) / len(client_mses)
avg_r2 = sum(client_r2s) / len(client_r2s)

print(f"\n Average (Global) MSE = {avg_mse:.2f}")
print(f" Average (Global) R² = {avg_r2 * 100:.2f}%")

end_time = time.time()
print(f"\n Total Execution Time: {end_time - start_time:.2f} seconds")


#--------------------plotting---------------------
import matplotlib.pyplot as plt

# Calculate average battery level per client
avg_battery_levels = []

for client_id in range(15):
    client_data = active_df.sample(frac=1/15, random_state=client_id)
    avg_battery_levels.append(client_data['battery_level'].mean())

# Plot battery_level vs. R²
plt.figure(figsize=(10, 6))
plt.plot(avg_battery_levels, [r * 100 for r in client_r2s], marker='o', linestyle='-', color='green')
plt.title("Client-wise Avg Battery Level vs. R² Score")
plt.xlabel("Average Battery Level (%)")
plt.ylabel("R² Score (%)")
plt.grid(True)
plt.tight_layout()
plt.show()

