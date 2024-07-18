from ultralytics import YOLO

import csv
from pathlib import Path
import os
import time

device_name = "CLUSTER"

# Data
# For COCO
root_dir = "./../../data/coco/val2017"
image_paths = list(Path(root_dir).glob("*.jpg"))

len_test_set = len(image_paths)
print(f"The test set has {len_test_set} instances.")

# Inference
filename = f"./../../output_data/Inference_Yolov8n-seg_COCO/{device_name}.csv"
os.makedirs(os.path.dirname(filename), exist_ok=True)

# Load a model
if device_name in ["RP4", "RP3"]:
    pre_model = YOLO(
        "./../../models/yolov8n-seg.pt"
    )  # load a pretrained model (recommended for training)
    pre_model.export(format="ncnn")
    model = YOLO("./../../models/yolov8n-seg_ncnn_model")
elif device_name in ["JETSONNANO"]:
    pre_model = YOLO("./../../models/yolov8n-seg.pt")
    pre_model.export(format="engine")  # creates 'yolov8n.engine'
    model = YOLO("./../../models/yolov8n-seg.engine")
else:
    model = YOLO("./../../models/yolov8n-seg.pt")

# Use the model
EPOCHS = 10
if device_name in ["RP3", "RP4", "JETSONNANO"]:
    BREAKPOINT = 100
elif device_name == "CLUSTER":
    BREAKPOINT = 1_000_000
else:
    BREAKPOINT = 1000
for epoch_number in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    for i in range(len(image_paths)):
        if i == BREAKPOINT:
            print(f"Break after {BREAKPOINT} images.")
            break
        image = image_paths[i]
        result = model.predict(
            image, stream=False, show_labels=False, show_conf=False, show_boxes=False
        )
        row = [1, result[0].speed["preprocess"], result[0].speed["inference"]]
        with open(filename, "a", newline="") as csvfile:
            csvwriter = csv.writer(
                csvfile, delimiter=",", quoting=csv.QUOTE_NONE, escapechar="\\"
            )
            csvwriter.writerow(row)
    epoch_end_time = time.time()
    epoch_time = epoch_end_time - epoch_start_time
    print(f"Finished Epoch {epoch_number} in {epoch_time} seconds")
