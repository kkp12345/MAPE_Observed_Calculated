import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load data from CSV
file_path = r"F:/_January2025/KanakPhD/Now040225/dacV_vgV_119k_0.978uf_Exp_22.csv"
df = pd.read_csv(file_path)

# Extract Va and Vg from the respective columns
Va = df.iloc[:, 3].values  # Measured Voltage
Vg = df.iloc[:, 1].values  # Generalized Voltage

# Define the voltage ranges for linear regression analysis
range_1_indices = (Vg >= 2.2) & (Vg <= 3.4)
range_2_indices = (Vg >= 3.4) & (Vg <= 4.2)
range_3_indices = (Vg >= 2.2) & (Vg <= 4.2)

# Data for range 1 (2.2V to  3.4V)
Vg_range_1 = Vg[range_1_indices].reshape(-1, 1)
Va_range_1 = Va[range_1_indices]

# Data for range 2 (3.5V to 4.2V)
Vg_range_2 = Vg[range_2_indices].reshape(-1, 1)
Va_range_2 = Va[range_2_indices]

# Data for range 2 (2.2V to 4.2V)
Vg_range_3 = Vg[range_3_indices].reshape(-1, 1)
Va_range_3 = Va[range_3_indices]

# Perform Linear Regression for both ranges
model_range_1 = LinearRegression().fit(Vg_range_1, Va_range_1)
model_range_2 = LinearRegression().fit(Vg_range_2, Va_range_2)
model_range_3 = LinearRegression().fit(Vg_range_3, Va_range_3)

# Get predictions for both ranges
Va_pred_range_1 = model_range_1.predict(Vg_range_1)
Va_pred_range_2 = model_range_2.predict(Vg_range_2)
Va_pred_range_3 = model_range_3.predict(Vg_range_3)


# Calculate Mean Absolute Percentage Error (MAPE) for both ranges
mape_range_1 = np.mean(np.abs((Va_range_1 - Va_pred_range_1) / Va_range_1)) * 100
mape_range_2 = np.mean(np.abs((Va_range_2 - Va_pred_range_2) / Va_range_2)) * 100
mape_range_3 = np.mean(np.abs((Va_range_3 - Va_pred_range_3) / Va_range_3)) * 100

# Print MAPE for both ranges
print("MAPE for range 2.2V to 3.4V: {:.2f}%".format(mape_range_1))
print("MAPE for range 3.4V to 4.2V: {:.2f}%".format(mape_range_2))
print("MAPE for range 2.2V to 4.2V: {:.2f}%".format(mape_range_3))

# Plot the data and linear regression fits for both ranges
plt.figure(figsize=(10, 6))
plt.plot(Vg_range_1, Va_range_1, 'bo', label="Data (2.2V to 3.4V)")
# plt.plot(Vg_range_1, Va_pred_range_1, 'r-', label="Fit (2.2V to just below 3.5V)")
plt.plot(Vg_range_2, Va_range_2, 'b*', label="Data (3.4V to 4.2V)", markersize=10)
# plt.plot(Vg_range_2, Va_pred_range_2, 'g-', label="Fit (3.5V to 4.2V)", markersize=12)
# plt.plot(Vg_range_3, Va_range_3, 'b*', label="Data (3.4V to 4.2V)", markersize=10)


# Add ideal straight line y = x
plt.plot([2.2, 4.2], [2.2, 4.2], 'k--', label="Ideal Line (y = x)", linewidth=2)

# Labels, title, and grid
plt.ylabel("Generalized Voltage (Vg) [V]")
plt.xlabel("Measured Voltage (Va) [V]")
# plt.title("Linear Regression Fit Comparison with Ideal Line","MAPE for range 2.2V to just below 3.4V: {:.2f}%".format(mape_range_1))
plt.title("Linear Regression Fit Comparison with Ideal Line")
    # "MAPE for range 2.2V to 3.4V: {:.2f}%\n"
    # "MAPE for range 3.4V to 4.2V: {:.2f}%\n"
    # "MAPE for range 2.2V to 4.2V: {:.2f}%".format(mape_range_1, mape_range_2,mape_range_3))
plt.legend()
plt.grid(True)
plt.show()
