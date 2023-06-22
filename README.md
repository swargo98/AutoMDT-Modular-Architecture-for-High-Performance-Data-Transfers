# Marlin: Judicious Parallelism During File Transfers
The application can only correctly function on Linux-based operating systems due to several Linux-based functional dependencies. There are two configuration files for specifying the source, destinations, maximum allowed thread, and many other options. Please use python3 and install the necessary packages using the requirements.txt file, preferably in a virtual environment, to avoid package version conflicts.


## Usage

1. Please create virtual environments on both the source and destination servers. For example: run `python3 -m venv <venv_dir>/marlin`
2. Activate the virtual environment: run `source <venv_dir>/marlin/bin/activate`
3. Install required Python packages: `pip3 install -r requirements.txt`
4. On the destination server, please edit `config_receiver.py` and run `python3 receiver.py`
5. On the source server, please edit `config_sender.py` and run `python3 sender.py`
