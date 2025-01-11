#!/bin/bash

source .venv/bin/activate
python rooporter.py
if [ "$1" == "shutdown_after_run" ]; then
    echo "shutdown"
    shutdown
fi
