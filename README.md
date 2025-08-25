# Federated Learning for UAV Battery Level Prediction

This repository contains my **final project** on applying **Federated Learning (FL)** techniques for predicting UAV battery levels using a real-world drone navigation dataset.  
The project also includes **data augmentation** for improving dataset size and model generalization.  

---

##  Project Overview
Unmanned Aerial Vehicles (UAVs) are widely used in navigation, surveillance, and delivery systems. Efficient **battery management** is critical for UAV operations.  
This project simulates a **Federated Learning setup with 15 clients** where each client trains a local model on a subset of UAV navigation data. The results are then aggregated to evaluate overall performance.  

Additionally, **data augmentation** is performed to expand the dataset and introduce variability for better learning.

---

##  Key Features
- **Data Augmentation**:  
  - Adds Gaussian noise (~3%) to numeric features.  
  - Doubles the dataset size while keeping non-numeric columns consistent.  

- **Federated Learning Simulation**:  
  - Splits data across 15 clients.  
  - Each client trains an **XGBoost Regressor** to predict UAV battery levels.  
  - Aggregates performance metrics (MSE, R²) across all clients.  

- **Feature Engineering**:  
  - Introduces a new `lat_long` feature (`latitude * longitude`).  
  - Filters out drones with battery below 40% for more meaningful learning.  

- **Visualization**:  
  - Plots **client-wise average battery level vs. R² score**.  

---

##  Project Structure
UAV-FL-Project

┣- data_augmentation.py # Script for creating augmented UAV data.

┣ federated_training.py # Main Federated Learning simulation script.

┣ augmented_uav_dataset.csv # The dataset generated after augmentation.

┣ README.md # This documentation file.

┗ requirements.txt # Python libraries needed to run the code.


## Tech Stack

Python 3.9+

XGBoost

Scikit-learn

Pandas, NumPy

Matplotlib

#Future Improvements

Implement true FedAvg aggregation for better Federated Learning simulation.

Extend the project to battery level classification (Low/Medium/High).

Deploy the model as a mobile app for real-time UAV monitoring.


## Author

Prerna Sharma

22cse090@gweca.ac.in

B.Tech CSE | Federated Learning | AI & ML Enthusiast
