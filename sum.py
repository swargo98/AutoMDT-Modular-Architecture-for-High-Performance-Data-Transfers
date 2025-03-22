import pandas as pd

# File paths for the two CSV files
file_marlin = "timed_log_network_ppo_marlin_write_bn.csv"
file_residual = "timed_log_network_ppo_automdt_write_bn.csv"

# Read CSV files without headers (adjust header or column names if needed)
df_marlin = pd.read_csv(file_marlin, header=None)
df_residual = pd.read_csv(file_residual, header=None)

# Sum the third column (index 2) for each DataFrame
sum_marlin = df_marlin.iloc[:, 2].sum()
sum_residual = df_residual.iloc[:, 2].sum()

# Calculate the total sum
total_sum = sum_marlin + sum_residual

print("Sum from ppo_marlin:", sum_marlin)
print("Sum from ppo_residual:", sum_residual)
print("Total sum:", total_sum)
