#!/usr/bin/env bash
source s0.3.4
PYTHONPATH=$PYTHONPATH:../../../ GLOG_vmodule=MemcachedClient=-1 \
spring.submit arun --gpu -p Test -n 1 -o log "python ../../../ptq/naive_ptq.py --config config.yaml"
