#!/bin/bash

# Data
cd data
mkdir coco
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip -d coco/

mkdir caltech256
cd caltech256
mkdir caltech256
cd caltech256
wget https://data.caltech.edu/records/nyy15-4j048/files/256_ObjectCategories.tar?download=1 -O 256_ObjectCategories.tar
cd ..
cd ..
cd ..

# Code
source .python-env/bin/activate
cd code
cd CNN_CIFAR10
./../.python-env/Scripts/python.exe Training.py
./../.python-env/Scripts/python.exe Inference.py
cd ..
cd CNN_FashionMNIST
./../.python-env/Scripts/python.exe Training.py
./../.python-env/Scripts/python.exe Inference.py