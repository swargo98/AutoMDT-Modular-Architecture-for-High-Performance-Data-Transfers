import pandas as pd
import matplotlib.pyplot as plt

# Define types for comparison
types = ["ppo_1", "ppo_2"]  # Add more types as needed

# Generate file paths dynamically
file_paths = {}
for t in types:
    file_paths[f"read_{t}"] = f"timed_log_read_{t}.csv"
    file_paths[f"network_{t}"] = f"timed_log_network_{t}.csv"
    file_paths[f"write_{t}"] = f"timed_log_write_{t}.csv"

# Load all data into a dictionary
data = {}
for key, path in file_paths.items():
    try:
        data[key] = pd.read_csv(path, header=None, names=["current_time", "time_since_beginning", "throughputs", "threads"])
    except FileNotFoundError:
        print(f"File not found: {path}")

# Function to create a plot with 3 subplots
def generate_plot(metric, x_axis, y_axis, ylabel, filename):
    fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=True)
    fig.suptitle(f"{ylabel} vs {x_axis.replace('_', ' ').capitalize()} ({metric.capitalize()})")

    for i, key in enumerate(["read", "network", "write"]):
        for t in types:
            df_key = f"{key}_{t}"
            if df_key in data:
                axes[i].plot(data[df_key][x_axis], data[df_key][y_axis], label=f"{t}", marker='o')
        axes[i].set_ylabel(f"{key.capitalize()} {ylabel}")
        axes[i].legend()

    axes[-1].set_xlabel(x_axis.replace("_", " ").capitalize())
    plt.tight_layout()
    plt.show()
    fig.savefig(filename)

# Generate and save all plots
for metric in ["throughputs", "threads"]:
    for x_axis in ["current_time", "time_since_beginning"]:
        ylabel = metric.capitalize()
        filename = f"competing_{metric}_vs_{x_axis}.png"
        generate_plot(metric, x_axis, metric, ylabel, filename)
