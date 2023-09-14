#!/bin/bash
# PTQ test
python ./mqbench/ptq_train_all_model.py

# QAT test
python ./mqbench/qat_train_all_model.py