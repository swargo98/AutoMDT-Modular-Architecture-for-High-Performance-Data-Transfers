import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data from the text files
continuous_rewards = np.loadtxt("continuous_rewards.txt", delimiter=",")
discrete_rewards = np.loadtxt("discrete_rewards.txt", delimiter=",")

# Convert the arrays to Pandas Series for rolling average calculation
cont_series = pd.Series(continuous_rewards)
disc_series = pd.Series(discrete_rewards)

# Compute a rolling average with a window of 100 points
cont_rolling = cont_series.rolling(window=100).mean()
disc_rolling = disc_series.rolling(window=100).mean()

# Create an index array (assuming each value corresponds to a sequential step)
indices = np.arange(len(continuous_rewards))

# Plot the rolling averages
plt.figure(figsize=(10, 6))
plt.plot(np.arange(len(continuous_rewards)), cont_rolling, label="Continuous Action Space", markersize=1)
plt.plot(np.arange(len(discrete_rewards)), disc_rolling, label="Discrete Action Space", markersize=1)

plt.xlabel("Episode")
plt.ylabel("Reward")
# plt.title("Comparison of Continuous and Discrete Rewards (100 Point Rolling Average)")
plt.legend()
plt.grid(True)
plt.savefig("reward_plots.pdf", dpi=300, format='pdf')
plt.show()