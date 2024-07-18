import torch
import torchvision
from torchvision import transforms

from PIL import Image

import time
import csv
from pathlib import Path
import os

device_name = "CLUSTER"

transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)


class CocoDataset(torchvision.datasets.VisionDataset):

    def load_image(self, index: int) -> Image:
        "Opens an image via a path and returns it."
        image_path = self.paths[index]
        img = Image.open(image_path)
        return img.convert("RGB")

    def __init__(self, root_dir, transform):
        # Let's buffer the underlying dataset, we will sample
        # from it on the fly
        self.paths = list(Path(root_dir).glob("*.jpg"))
        self.transform = transform

    def __len__(self):
        # Typical front dataset : size is the same as the
        # underlying dataset size
        return len(self.paths)

    def __getitem__(self, index):
        # sampling from CIFAR10
        img = self.load_image(index)
        # Because you want to return a list
        return self.transform(img)


TEST_BATCHSIZE = 1
test_set = CocoDataset("./../../data/coco/val2017", transform=transform)
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=TEST_BATCHSIZE, shuffle=True
)

len_test_set = len(test_set)
print(f"The test set has {len_test_set} instances.")


# CUDA or CPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using {str(device).upper()}!\n")

if device_name in ["JETSONNANO"]:
    from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn

    model = fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
else:
    from torchvision.models.detection import (
        fasterrcnn_mobilenet_v3_large_320_fpn,
        FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    )

    weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)

model.eval()
model.to(device)


filename = f"./../../output_data/Inference_FasterRCNN_COCO/{device_name}.csv"
os.makedirs(os.path.dirname(filename), exist_ok=True)

# edge_device = False
# if device_name == "RP3" or device_name == "RP4":
#     edge_device = True

EPOCHS = 3
if device_name in ["RP3", "RP4", "JETSONNANO"]:
    BREAKPOINT = 100
elif device_name == "CLUSTER":
    BREAKPOINT = 1_000_000
else:
    BREAKPOINT = 1000

for epoch_number in range(1, EPOCHS + 1):
    print(f"  Starting EPOCH {epoch_number}:")
    with torch.no_grad():
        epoch_start_time = time.time()
        for i, data in enumerate(test_loader):
            if i == BREAKPOINT:
                print(f"  Break after {BREAKPOINT} images.")
                break
            inputs = data[0].to(device)
            outputs = model(inputs.unsqueeze(0))
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time

        row = [device_name, BREAKPOINT, TEST_BATCHSIZE, epoch_time]
        with open(filename, "a", newline="") as csvfile:
            csvwriter = csv.writer(
                csvfile, delimiter=",", quoting=csv.QUOTE_NONE, escapechar="\\"
            )
            csvwriter.writerow(row)

    print(f"  Finished Epoch {epoch_number} in {epoch_time} seconds!")
