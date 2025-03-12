## Falcon Configurations

configurations = {
    "receiver": {
        "host": "10.10.1.2",
        "port": 50028
    },
    "rpc_port":"5002",
    "data_dir": "src/",
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
    "network_limit": 60, # Network limit (Mbps) per thread
    "io_limit": 150, # I/O limit (Mbps) per thread
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
    'competing_transfer': 0,
    # "mp_opt": True,
    # "method": "ppo",
    "mp_opt": False,
    "method": "gradient", # options: [gradient, bayes, random, brute, probe, cg, lbfgs]
    "multiplier": 1, # multiplier for each files, only for testing purpose
    "model_version": 'marlin_network_bn',
    "mode": 'inference',
    'inference_value_model': 'read_bn_finetune_value_150.pth',
    'inference_policy_model': 'read_bn_finetune_policy_150.pth',
    'finetune_value_model': 'network_bn_offline_value_12300.pth',
    'finetune_policy_model': 'network_bn_offline_policy_12300.pth',
    'max_episodes': 151,
}
