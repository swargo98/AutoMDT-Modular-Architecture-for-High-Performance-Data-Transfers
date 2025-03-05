## Falcon Configurations

configurations = {
    "receiver": {
        "host": "10.10.1.2",
        "port": 50026
    },
    "rpc_port":"5002",
    "data_dir": "src/",
    # "method": "bayes", # options: [gradient, bayes, random, brute, probe, cg, lbfgs]
    "method": "ppo",
    "bayes": {
        "initial_run": 3,
        "num_of_exp": -1 #-1 for infinite
    },
    "random": {
        "num_of_exp": 10
    },
    "centralized": False, # True for centralized optimization
    "file_transfer": True,
    "B": 10, # severity of the packet loss punishment
    "K": 1.05, # cost of increasing concurrency
    "loglevel": "info",
    "probing_sec": 3, # probing interval in seconds
    "multiplier": 50, # multiplier for each files, only for testing purpose
    "mp_opt": True,
    "network_limit": 100, # Network limit (Mbps) per thread
    "io_limit": 333, # I/O limit (Mbps) per thread
    "memory_use": {
        "maximum": 5,
        "threshold": 1,
    },
    "fixed_probing": {
        "bsize": 10,
        "thread": 3
    },
    "max_cc": {
        "network": 20,
        "io": 20,
        'write': 20
    },
    "model_version": '9',
}
