#!/bin/bash
#TODO options for service
# see https://bert-as-service.readthedocs.io/en/latest/source/server.html
bert-serving-start -model_dir /tmp/english_L-12_H-768_A-12/ -num_worker=2
