import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------------------------
# 1. Define model types and bottlenecks
# -------------------------------------------------
types = ["ppo_automdt", "ppo_marlin"]  # Two model types
bottlenecks = ["full"]

# Subplot titles for bottlenecks
bottleneck_labels = {
    "no_bn":    "Regular Transfer",
    "full":     "Regular Transfer",
    "read_bn":  "Read I/O Bottleneck",
    "network_bn": "Network Bottleneck",
    "write_bn": "Write I/O Bottleneck"
}

# Model labels for legend and y-axis labeling
type_labels = {
    "ppo_automdt": "AutoMDT",
    "ppo_marlin":  "Marlin",
}

# -------------------------------------------------
# 2. Define link speeds (throughput per thread) for each bottleneck
# -------------------------------------------------
link_speeds = {
    "no_bn": {"read": 775, "network": 215,  "write": 335},
    "full":  {"read": 1950, "network": 1480,  "write": 1810},
    "network_bn": {"read": 265, "network": 90,  "write": 220},
    "read_bn":    {"read": 70,  "network": 160, "write": 170},
    "write_bn":   {"read": 265, "network": 160, "write": 70},
}

# -------------------------------------------------
# 3. Generate file paths dynamically
#    Code loads all CSVs: read, network, and write files for each type and bottleneck.
# -------------------------------------------------
file_paths = {}
for t in types:
    for b in bottlenecks:
        file_paths[f"read_{t}_{b}"]    = f"logs_full/timed_log_read_{t}_{b}.csv"
        file_paths[f"network_{t}_{b}"] = f"logs_full/timed_log_network_{t}_{b}.csv"
        file_paths[f"write_{t}_{b}"]   = f"logs_full/timed_log_write_{t}_{b}.csv"

# -------------------------------------------------
# 4. Load CSV data into a dictionary
# -------------------------------------------------
data = {}
for key, path in file_paths.items():
    try:
        df = pd.read_csv(
            path, 
            header=None, 
            names=["current_time", "time_since_beginning", "throughputs", "threads"]
        )
        data[key] = df
    except FileNotFoundError:
        print(f"File not found: {path}")

# -------------------------------------------------
# 5. Define common line styles for the concurrency (threads) plots.
# -------------------------------------------------
line_styles = {
    "read":    {"linestyle": "--", "color": "red"},
    "network": {"linestyle": "-.", "color": "green"},
    "write":   {"linestyle": ":",  "color": "blue"},
}

# -------------------------------------------------
# 6. Precompute common x-axis and y-axis limits for concurrency plots per bottleneck.
# -------------------------------------------------
common_xlim = {}  # common x-axis (data length) for each bottleneck
common_ylim = {}  # common y-axis limits (min and max of rolling "threads") for each bottleneck

for b in bottlenecks:
    max_xlim = 0
    global_min = float('inf')
    global_max = float('-inf')
    # Iterate over each model type and each line type for the current bottleneck.
    for t in types:
        for sub_key in ["read", "network", "write"]:
            dict_key = f"{sub_key}_{t}_{b}"
            if dict_key in data:
                current_length = len(data[dict_key].index)
                if current_length > max_xlim:
                    max_xlim = current_length
                # Compute the 5-point rolling average for "threads".
                conv_roll = data[dict_key]["threads"].rolling(window=5).mean()
                current_min = conv_roll.min()
                current_max = conv_roll.max()
                if pd.notna(current_min) and current_min < global_min:
                    global_min = current_min
                if pd.notna(current_max) and current_max > global_max:
                    global_max = current_max
    common_xlim[b] = max_xlim
    if global_min == float('inf'):
        global_min = 0
    common_ylim[b] = (global_min, global_max)

# -------------------------------------------------
# 7. Precompute common x-axis and y-axis limits for network throughput plots per bottleneck.
# -------------------------------------------------
common_tp_xlim = {}  # common x-axis (number of data points) for each bottleneck
common_tp_ylim = {}  # common y-axis (min and max throughputs) for each bottleneck

for b in bottlenecks:
    max_xlim = 0
    global_min = float('inf')
    global_max = float('-inf')
    for t in types:
        key = f"network_{t}_{b}"
        if key in data:
            current_length = len(data[key].index)
            if current_length > max_xlim:
                max_xlim = current_length
            # Compute the 5-point rolling average for the "throughputs" column.
            tp_roll = data[key]["throughputs"].rolling(window=5).mean()
            current_min = tp_roll.min()
            current_max = tp_roll.max()
            if pd.notna(current_min) and current_min < global_min:
                global_min = current_min
            if pd.notna(current_max) and current_max > global_max:
                global_max = current_max
    common_tp_xlim[b] = max_xlim
    if global_min == float('inf'):
        global_min = 0
    common_tp_ylim[b] = (global_min, global_max)

# -------------------------------------------------
# 8. Create and save separate figures for concurrency plots.
#    (Total: one per (model type, bottleneck) pair)
# -------------------------------------------------
for t in types:
    for b in bottlenecks:
        fig, ax = plt.subplots()
        # Plot each of the three lines: read, network, write.
        for sub_key in ["read", "network", "write"]:
            dict_key = f"{sub_key}_{t}_{b}"
            if dict_key in data:
                # Compute a 5-point rolling average for the "threads" column.
                conv_roll = data[dict_key]["threads"].rolling(window=5).mean()
                avg_threads = data[dict_key]["threads"].mean()
                print(f"{dict_key}: Average threads = {avg_threads:.2f}")
                speeds_for_bottleneck = link_speeds[b]
                label_str = f"{sub_key.capitalize()} = {speeds_for_bottleneck[sub_key]} Mbps"
                ax.plot(conv_roll.index, conv_roll, label=label_str, **line_styles.get(sub_key, {}))
        
        # Set plot title and axis labels.
        ax.set_title(f"{bottleneck_labels[b]} - Concurrency\n{type_labels.get(t, t)}")
        ax.set_ylabel(f"Concurrency ({type_labels.get(t, t)})")
        ax.set_xlabel("Duration (Seconds)")
        
        # Set common x-axis.
        # max_xlim_val = common_xlim[b]
        # ax.set_xlim(0, max_xlim_val)
        # # Use 5 evenly spaced ticks.
        # ax.set_xticks(np.linspace(0, max_xlim_val, 5, dtype=int))
        
        # # Set common y-axis.
        # ymin, ymax = common_ylim[b]
        # ax.set_ylim(ymin, ymax+5)
        # ax.set_yticks(np.linspace(ymin, ymax+5, 5))

        max_xlim_val = int(common_xlim[b])
        ax.set_xlim(0, max_xlim_val)
        step_x = max(1, max_xlim_val // 4)  # aiming for 5 ticks (including 0 and max)
        ax.set_xticks(range(0, max_xlim_val + 1, step_x))

        # For the y-axis ticks:
        tp_ymin, tp_ymax = common_ylim[b]
        tp_ymin_int = int(tp_ymin)
        tp_ymax_int = int(tp_ymax)
        step_y = max(1, (tp_ymax_int - tp_ymin_int) // 4)  # aiming for 5 ticks
        ax.set_yticks(range(tp_ymin_int, tp_ymax_int + 1, step_y))
    
        
        ax.grid(True)
        ax.legend(fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f"subplots/concurrency_{t}_{b}.pdf", dpi=300, format='pdf')
        plt.show()

# -------------------------------------------------
# 9. Create and save separate figures for network throughput plots.
#    (One plot per bottleneck combining all model types)
# -------------------------------------------------
for b in bottlenecks:
    fig, ax = plt.subplots()
    for t in types:
        key = f"network_{t}_{b}"
        if key in data:
            # Compute a 5-point rolling average for the "throughputs" column.
            tp_roll = data[key]["throughputs"].rolling(window=5).mean()
            label_str = f"{type_labels[t]}"
            ax.plot(tp_roll.index, tp_roll, label=label_str)
    
    # Set plot title and axis labels.
    ax.set_title(f"{bottleneck_labels[b]} - Throughput")
    ax.set_ylabel("Throughput (Mbps)")
    ax.set_xlabel("Duration (Seconds)")
    
    # Set common x-axis.
    # max_tp_xlim = common_tp_xlim[b]
    # ax.set_xlim(0, max_tp_xlim)
    # ax.set_xticks(np.linspace(0, max_tp_xlim, 5, dtype=int))
    
    # # Set common y-axis.
    # tp_ymin, tp_ymax = common_tp_ylim[b]
    # ax.set_ylim(tp_ymin, tp_ymax+500)
    # tick_gap = int(int(int(int(tp_ymax)+500//5)//500)*500)
    # ax.set_yticks(range(tp_ymin, tp_ymax+500, tick_gap))

    max_tp_xlim = int(common_tp_xlim[b])
    ax.set_xlim(0, max_tp_xlim)
    step_x = max(1, max_tp_xlim // 4)  # aiming for 5 ticks (including 0 and max)
    ax.set_xticks(range(0, max_tp_xlim + 1, step_x))

    # For the y-axis ticks:
    tp_ymin, tp_ymax = common_tp_ylim[b]
    tp_ymin_int = int(tp_ymin)
    tp_ymax_int = int(tp_ymax)
    step_y = max(1, (tp_ymax_int - tp_ymin_int) // 4)  # aiming for 5 ticks
    ax.set_yticks(range(tp_ymin_int, tp_ymax_int + 1, step_y))
    
    ax.grid(True)
    ax.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f"subplots/throughput_{b}.pdf", dpi=300, format='pdf')
    plt.show()