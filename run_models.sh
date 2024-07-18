#!/bin/bash

# Code
cd code
cd CNN_CIFAR10
../.python-env/bin/python Training.py
../.python-env/bin/python Inference.py
cd ..
cd CNN_FashionMNIST
../.python-env/bin/python Training.py
../.python-env/bin/python Inference.py
# etc