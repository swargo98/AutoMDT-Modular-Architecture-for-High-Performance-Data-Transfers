import matplotlib.pyplot as plt

# Define types for comparison
types = ["rnw", "ppo"]  # You can add more types (e.g., ["rnw", "ppo", "adam"])

# Generate file paths dynamically
file_paths = {}
for t in types:
    file_paths[f"threads_{t}"] = f"threads_log_{t}.csv"
    file_paths[f"throughputs_{t}"] = f"throughputs_log_{t}.csv"

# Load data from files
data = {}
for key, path in file_paths.items():
    with open(path, "r") as file:
        lines = file.readlines()
    data[key] = lines

def parse_data(lines):
    """Parses the given lines into read, network, and write components."""
    read, network, write = [], [], []
    
    for i, line in enumerate(lines):
        values = list(map(float, line.strip("[]\n").split(',')))
        
        if i % 2 == 0:  # Odd lines (0-based index) contain read and network values
            read.append(values[0])
            network.append(values[1] if len(values) > 1 else 0)
        else:  # Even lines contain write values
            write.append(values[0])

    return read, network, write

# Parse the data for all types
parsed_data = {}
for t in types:
    parsed_data[f"threads_{t}"] = parse_data(data[f"threads_{t}"])
    parsed_data[f"throughputs_{t}"] = parse_data(data[f"throughputs_{t}"])

# Generate comparison plots function
def generate_plots(metric, ylabel, filename):
    fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=True)
    fig.suptitle(f"{metric.capitalize()} Comparison")

    for t in types:
        read, network, write = parsed_data[f"{metric}_{t}"]
        axes[0].plot(read, label=f"{t} Read", marker='o')
        axes[1].plot(network, label=f"{t} Network", marker='s')
        axes[2].plot(write, label=f"{t} Write", marker='^')

    axes[0].set_ylabel("Read")
    axes[1].set_ylabel("Network")
    axes[2].set_ylabel("Write")
    axes[2].set_xlabel("Index")

    for ax in axes:
        ax.legend()

    plt.show()
    fig.savefig(filename)

# Generate and save plots
generate_plots("threads", "Threads", "threads_comparison.png")
generate_plots("throughputs", "Throughput", "throughput_comparison.png")