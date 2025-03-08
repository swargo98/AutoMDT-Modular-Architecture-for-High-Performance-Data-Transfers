## Falcon Configurations

configurations = {
    "receiver": {
        "host": "10.10.1.2",
        "port": 50026
    },
    "rpc_port":"5002",
    "data_dir": "src/",
    # "mp_opt": True,
    # "method": "ppo",
    "mp_opt": True,
    "method": "mgd", # options: [gradient, bayes, random, brute, probe, cg, lbfgs]
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
    "multiplier": 1, # multiplier for each files, only for testing purpose
    "model_version": 'gradient_mgd',
    "mode": 'inference',
    'inference_value_model': 'training_dicrete_w_history_minibatch_mlp_deepseek_v12_value_400000.pth',
    'inference_policy_model': 'training_dicrete_w_history_minibatch_mlp_deepseek_v12_policy_400000.pth',
}
