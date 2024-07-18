import torch
import torchvision
from torchvision import transforms

import time
import csv
import os

device_name = "CLUSTER"

transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

TEST_BATCHSIZE = 1
test_set = torchvision.datasets.Caltech256(
    "./../../data/caltech256", transform=transform, download=True
)
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=TEST_BATCHSIZE, shuffle=False
)

len_test_set = len(test_set)
print(f"The test set has {len_test_set} instances.")


# CUDA or CPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using {str(device).upper()}!\n")
if device_name in ["JETSONNANO"]:
    from torchvision.models import resnet50

    model = resnet50(pretrained=True)
else:
    from torchvision.models import resnet50, ResNet50_Weights

    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
model.to(device)
model.eval()
# Load trained model from training here?


filename = f"./../../output_data/Inference_Resnet50_Caltech256/{device_name}.csv"
os.makedirs(os.path.dirname(filename), exist_ok=True)

EPOCHS = 10
if device_name in ["RP3", "RP4", "JETSONNANO"]:
    BREAKPOINT = 1000
elif device_name == "CLUSTER":
    BREAKPOINT = 1_000_000
else:
    BREAKPOINT = 10000

for epoch_number in range(1, EPOCHS + 1):
    print(f"  Starting EPOCH {epoch_number}:")

    epoch_start_time = time.time()
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            if i == BREAKPOINT:
                print(f"Break after {BREAKPOINT} images.")
                break
            inputs = data[0].to(device)
            outputs = model(inputs)
    epoch_end_time = time.time()
    epoch_time = epoch_end_time - epoch_start_time

    row = [device_name, BREAKPOINT, TEST_BATCHSIZE, epoch_time]
    with open(filename, "a", newline="") as csvfile:
        csvwriter = csv.writer(
            csvfile, delimiter=",", quoting=csv.QUOTE_NONE, escapechar="\\"
        )
        csvwriter.writerow(row)

    print(f"  Finished Epoch {epoch_number} in {epoch_time} seconds!")
