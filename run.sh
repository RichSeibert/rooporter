#!/bin/bash

# TODO figure out a way to download apt stuff to a non-volitile directory, so it's
# not deleted when starting a new instance. I shouldn't need to run setup.sh 
# every time
echo "Run setup.sh"
bash setup.sh

echo "Run rooporter.py"
source .venv/bin/activate
python rooporter.py
