import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------
# 1. Define model types and bottleneck
# -------------------------------------------------
types = ["ppo_automdt", "ppo_automdt_wo_ft"]
type_labels = {
    "ppo_automdt":       "AutoMDT FT",
    "ppo_automdt_wo_ft": "AutoMDT Offline",
}

# We are focusing on the 'network_bn' scenario
bottleneck = "network_bn"
bottleneck_labels = {
    "network_bn": "Network Bottleneck"
}

# Define link speeds (throughput per thread in Mbps) for the 'network_bn' bottleneck.
link_speeds = {
    "network_bn": {"read": 265, "network": 90, "write": 220},
}

# Line styles for each data type
line_styles = {
    "read":    {"linestyle": "--", "color": "red"},
    "network": {"linestyle": "-.", "color": "green"},
    "write":   {"linestyle": ":",  "color": "blue"},
}

# -------------------------------------------------
# 2. Define file paths for CSV files and load data
# -------------------------------------------------
# We assume that the CSVs have columns: current_time, time_since_beginning, throughputs, threads
# and are stored in a folder called "temp" with naming:
# "timed_log_{sub_key}_{model_type}_{bottleneck}.csv" where sub_key is one of "read", "network", "write"
file_paths = {}
for t in types:
    for sub_key in ["read", "network", "write"]:
        key = f"{sub_key}_{t}_{bottleneck}"
        file_paths[key] = f"temp/timed_log_{sub_key}_{t}_{bottleneck}.csv"

# Load CSV data into a dictionary; if not found, print a message.
data = {}
for key, path in file_paths.items():
    try:
        df = pd.read_csv(
            path,
            header=None,
            names=["current_time", "time_since_beginning", "throughputs", "threads"]
        )
        data[key] = df
        print(f"Loaded file: {path}")
    except FileNotFoundError:
        print(f"File not found: {path}")

# -------------------------------------------------
# 3. Create a figure with 2 subplots (one per model type)
# -------------------------------------------------
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 6), sharex=True, sharey=True)

# -------------------------------------------------
# 4. Populate each subplot with concurrency (threads) plots
# -------------------------------------------------
for row_idx, t in enumerate(types):
    ax = axes[row_idx]
    
    for sub_key in ["read", "network", "write"]:
        dict_key = f"{sub_key}_{t}_{bottleneck}"
        
        if dict_key in data:
            # Compute 5-point rolling average for the "threads" column
            rolling_threads = data[dict_key]["threads"].rolling(window=5).mean()
            
            # Build label string to include link speed information
            speed = link_speeds[bottleneck][sub_key]
            label_str = f"{sub_key.capitalize()} = {speed} Mbps"
            
            ax.plot(
                rolling_threads.index,
                rolling_threads,
                label=label_str,
                **line_styles[sub_key]
            )
        else:
            print(f"Data for {dict_key} is missing.")

    # Set subplot title for the top row only
    if row_idx == 0:
        ax.set_title(bottleneck_labels[bottleneck], fontsize=12)
    
    ax.set_ylabel(f"Concurrency\n({type_labels[t]})")
    ax.set_xlabel("Duration (Seconds)")
    
    # Set y-axis limits (adjust if necessary)
    ax.set_ylim(0, 21)
    ax.set_yticks(range(0, 22, 5))
    
    ax.grid(True)
    ax.legend(fontsize=9, loc="upper right")

# -------------------------------------------------
# 5. Final layout adjustments and show/save the plot
# -------------------------------------------------
plt.tight_layout()
plt.savefig("concurrency_network_bn.pdf", dpi=300, format='pdf')
plt.show()