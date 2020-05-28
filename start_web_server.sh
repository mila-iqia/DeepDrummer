#!/usr/bin/env bash

export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export OMP_NUM_THREADS=4

# export FLASK_RUN_PORT=5000
export FLASK_DEBUG=1
export FLASK_APP=deepdrummer.webserver.webserver
flask run --host=0.0.0.0
