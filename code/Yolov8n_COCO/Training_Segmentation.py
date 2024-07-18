from ultralytics import YOLO

import csv
import os
import time

device_name = "CLUSTER"
# Training
filename = f"./../../output_data/Training_Yolov8n-seg_COCO/{device_name}.csv"
os.makedirs(os.path.dirname(filename), exist_ok=True)

model = YOLO("./../../models/yolov8n-seg.pt")

BATCHSIZES = [1,4,16,64,256]
EPOCHS = 10

for batchsize in BATCHSIZES:
    training_start_time = time.time()
    
    results = model.train(data="./../../data/coco-yolo/coco-seg.yaml", epochs=EPOCHS, imgsz=640, exist_ok=True, batch=batchsize)
    
    training_end_time = time.time()
    training_time = training_end_time - training_start_time

    row = ["COCO-SEG", training_time, batchsize, EPOCHS]
    with open(filename, "a", newline="") as csvfile:
        csvwriter = csv.writer(
            csvfile, delimiter=",", quoting=csv.QUOTE_NONE, escapechar="\\"
        )
        csvwriter.writerow(row)

    print(f"FinishedTraining with Batchsize {batchsize} in {training_time} seconds")
