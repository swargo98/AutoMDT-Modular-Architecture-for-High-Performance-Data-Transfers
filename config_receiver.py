configurations = {
    "receiver": {
        "host": "127.0.0.1",
        "port": 50022
    },
    "data_dir": "dest/",
    "method": "gradient", # options: [gradient, bayes, random, brute, probe, cg, lbfgs]
    "bayes": {
        "initial_run": 3,
        "num_of_exp": -1 #-1 for infinite
    },
    "max_cc": 15,
    "K": 1.02,
    "probing_sec": 3, # probing interval in seconds
    "file_transfer": True,
    "io_limit": 80, # I/O limit (Mbps) per thread
    "loglevel": "info",
}