#!/usr/bin/env bash
PYTHONPATH=$PYTHONPATH:../../../ GLOG_vmodule=MemcachedClient=-1 \
spring.submit arun --gpu -p ToolChain -n 1 -o log "python ../../../ptq/advanced_ptq.py --config config.yaml"
