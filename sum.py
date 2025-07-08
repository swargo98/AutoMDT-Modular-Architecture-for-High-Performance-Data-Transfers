import pandas as pd

# File paths for the two CSV files
file_marlin = "timed_log_write_ppo_marlin_full.csv"
file_residual = "timed_log_write_ppo_automdt_full.csv"
file_fixed = "timed_log_write_ppo_fixed_5.csv"

# Read CSV files without headers (adjust header or column names if needed)
df_marlin = pd.read_csv(file_marlin, header=None)
df_residual = pd.read_csv(file_residual, header=None)
df_fixed = pd.read_csv(file_fixed, header=None)

# Sum the third column (index 2) for each DataFrame
sum_marlin = df_marlin.iloc[:, 2].sum()
sum_residual = df_residual.iloc[:, 2].sum()
sum_fixed = df_fixed.iloc[:, 2].sum()

# get the highest value from the second column (index 1) for each DataFrame
max_marlin = df_marlin.iloc[:, 1].max()
max_residual = df_residual.iloc[:, 1].max()
max_fixed = df_fixed.iloc[:, 1].max()

print("Sum from ppo_marlin:", sum_marlin)
print("Sum from ppo_residual:", sum_residual)
print("Sum from ppo_fixed:", sum_fixed)

# the sum is throughput and max is time. Print how much time it takes to transfer 1TB
print("Time to transfer 1TB with ppo_marlin:", 10000000 / sum_marlin * max_marlin, "seconds")
print("Time to transfer 1TB with ppo_residual:", 10000000 / sum_residual * max_residual, "seconds")
print("Time to transfer 1TB with ppo_fixed:", 10000000 / sum_fixed * max_fixed, "seconds")
