#!/bin/sh
srun --partition=Test --kill-on-bad-exit=1 -n1 --gres=gpu:1 --ntasks-per-node=1 --mpi=pmi2 --quotatype=reserved python ../../../ptq/naive_ptq.py --config config.yaml
