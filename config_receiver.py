configurations = {
    "receiver": {
        "host": "10.10.1.2",
        "port": 50026
    },
    "rpc_port":"5002",
    "data_dir": "dest/",
    "method": "ppo", # options: [gradient, bayes, random, brute, probe, cg, lbfgs]
    # "method": "ppo",
    "bayes": {
        "initial_run": 3,
        "num_of_exp": -1 #-1 for infinite
    },
    "max_cc": 20,
    "K": 1.02,
    "probing_sec": 3, # probing interval in seconds
    "file_transfer": True,
    "io_limit": 333, # I/O limit (Mbps) per thread
    "loglevel": "info",
    "model_version": '12',
}