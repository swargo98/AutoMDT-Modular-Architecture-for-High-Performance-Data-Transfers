import os

# Define the input files for read, network, and write
input_files = {
    "read": "threads_log_univ_gd_read.csv",     # Replace with the actual path
    "network": "threads_log_univ_gd_network.csv",
    "write": "threads_log_univ_gd_write.csv",
}

# Define the output file
output_file = "threads_log_univ_gd.csv"

def combine_files(read_file, network_file, write_file, output_file):
    """
    Combines read, network, and write files into a single alternating structure.
    Args:
        read_file (str): Path to the read file.
        network_file (str): Path to the network file.
        write_file (str): Path to the write file.
        output_file (str): Path to the output combined file.
    """
    with open(read_file, "r") as rf, open(network_file, "r") as nf, open(write_file, "r") as wf, open(output_file, "w") as of:
        # Read all lines from each file
        read_lines = rf.readlines()
        network_lines = nf.readlines()
        write_lines = wf.readlines()

        # Iterate over the lines and write to the output file
        for i in range(len(read_lines)):
            # Combine read and network (remove any whitespace or newlines)
            read = read_lines[i].strip()
            network = network_lines[i].strip()
            of.write(f"{read}, {network}\n")  # Odd lines: read, network

            # Write the corresponding write value
            if i < len(write_lines):
                write = write_lines[i].strip()
                of.write(f"{write}\n")  # Even lines: write

# Call the function
combine_files(input_files["read"], input_files["network"], input_files["write"], output_file)

print(f"Combined file saved to {output_file}")
