## Falcon Configurations

configurations = {
    "receiver": {
        "host": "192.168.1.1",
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
    "K": 1.02, # cost of increasing concurrency
    "loglevel": "info",
    "probing_sec": 3, # probing interval in seconds
    "network_limit": -1, # Network limit (Mbps) per thread
    "io_limit": -1, # I/O limit (Mbps) per thread
    "memory_use": {
        "maximum": 10,
        "threshold": 1,
    },
    "fixed_probing": {
        "bsize": 10,
        "thread": 3
    },
    "max_cc": {
        "network": 100,
        "io": 100,
        'write': 100
    },
    'competing_transfer': 0,
    # "mp_opt": True,
    # "method": "ppo",
    "mp_opt": False,
    "method": "gradient", # options: [gradient, bayes, random, brute, probe, cg, lbfgs]
    "multiplier": 2, # multiplier for each files, only for testing purpose
    'max_episodes': 120,
    "model_version": 'marlin_4gb',
    "mode": 'inference',
    'inference_value_model': 'best_models/automdt_4gb_finetune_value.pth',
    'inference_policy_model': 'best_models/automdt_4gb_finetune_policy.pth',
    'finetune_value_model': 'best_models/automdt_network_bn_offline_value.pth',
    'finetune_policy_model': 'best_models/automdt_network_bn_offline_policy.pth',
}
