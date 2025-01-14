#!/bin/bash

echo "Run setup.sh"
bash setup.sh

echo "Run rooporter.py"
source .venv/bin/activate
python rooporter.py
