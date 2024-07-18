#!/bin/bash

mkdir data
mkdir models
mkdir output_data

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

cd data

mkdir coco-yolo
cp ../setup_data/coco-seg.yaml coco-yolo

mkdir voc
cp ../setup_data/VOC.yaml voc

cd ..