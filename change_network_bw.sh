sudo tc qdisc del dev enp7s0 root

sudo tc qdisc add dev enp7s0 root handle 1: htb default 1

sudo tc class add dev enp7s0 parent 1: classid 1:1 htb rate 1000mbit ceil 1000mbit