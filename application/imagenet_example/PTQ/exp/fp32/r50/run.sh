#!/usr/bin/env bash
source s0.3.4
PYTHONPATH=$PYTHONPATH:../../../ GLOG_vmodule=MemcachedClient=-1 \
spring.submit arun --gpu -n 1 -p ToolChain -o log "python ../../../ptq/naive_ptq.py --config config.yaml"
