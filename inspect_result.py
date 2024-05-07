# %%
import os
import numpy as np

# %%
model_dir = "./results"
model_name = ""
metrics_file = os.path.join(model_dir, model_name, "metrics.npy")

metrics = np.load(metrics_file)
indexes = [1, 0, 2, 3, 4]

print(f"MAE: {metrics[0]:.6f}")
print(f"MSE: {metrics[1]:.6f}")
print(f"RMSE: {metrics[2]:.6f}")
print(f"MAPE: {metrics[3]:.6f}")
print(f"MSPE: {metrics[4]:.6f}")

print("\t".join([f"{metrics[ind]:.4f}" for ind in indexes]) + "\n")

# %%
