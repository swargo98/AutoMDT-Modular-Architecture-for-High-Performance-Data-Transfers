configurations = {
    "receiver": {
        "host": "10.10.1.2",
        "port": 50028
    },
    "rpc_port":"5002",
    "data_dir": "dest/",
    "bayes": {
        "initial_run": 3,
        "num_of_exp": -1 #-1 for infinite
    },
    "max_cc": 20,
    "K": 1.02,
    "probing_sec": 3, # probing interval in seconds
    "file_transfer": True,
    "loglevel": "info",
    "io_limit": 150, # I/O limit (Mbps) per thread
    'competing_transfer': 0,
    # "method": "ppo",
    "method": "gradient", # options: [gradient, bayes, random, brute, probe, cg, lbfgs]
    "model_version": 'residual',
}