#!/usr/bin/env bash
PYTHONPATH=$PYTHONPATH:../../../ GLOG_vmodule=MemcachedClient=-1 \
spring.submit arun --gpu -n 1 -p Test -o log "python ../../../ptq/naive_ptq.py --config config.yaml"
