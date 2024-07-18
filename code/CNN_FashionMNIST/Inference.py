import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

import time
import csv
import os

device_name = "CLUSTER"

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

# Create datasets for training & validation, download if necessary
TEST_BATCHSIZE = 1
test_set_short = torchvision.datasets.FashionMNIST(
    "./../../data", train=False, transform=transform, download=True
)
test_loader_short = torch.utils.data.DataLoader(
    test_set_short, batch_size=TEST_BATCHSIZE, shuffle=True
)

test_set_long = torchvision.datasets.FashionMNIST(
    "./../../data", train=True, transform=transform, download=True
)
test_loader_long = torch.utils.data.DataLoader(
    test_set_long, batch_size=TEST_BATCHSIZE, shuffle=True
)

# Report split sizes
len_test_set_short = len(test_set_short)
len_test_set_long = len(test_set_long)
print(f"Short test set has {len_test_set_short} instances")
print(f"Long test set has {len_test_set_long} instances")

classes = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle Boot",
)


# PyTorch models inherit from torch.nn.Module
class GarmentClassifier(nn.Module):
    def __init__(self):
        super(GarmentClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using {str(device).upper()}!\n")

model = GarmentClassifier()
model_path = "./../../models/CNN_FashionMNIST_trained_model"
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()


filename = f"./../../output_data/Inference_CNN_FashionMNIST/{device_name}.csv"
os.makedirs(os.path.dirname(filename), exist_ok=True)

EPOCHS = 3
if device_name in ["RP3", "RP4", "JETSONNANO"]:
    BREAKPOINT = 1000
else:
    BREAKPOINT = 10000

for epoch_number in range(1, EPOCHS + 1):
    print(f"Starting Epoch {epoch_number}:")

    # SHORT TESTSET
    epoch_start_time = time.time()
    with torch.no_grad():
        for i, data in enumerate(test_loader_short):
            if i == BREAKPOINT:
                print(f"  Break after {BREAKPOINT} images.")
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
    print(f"  Finished short dataset {epoch_number} in {epoch_time} seconds!")

    # LONG TESTSET
    epoch_start_time = time.time()
    with torch.no_grad():
        for i, data in enumerate(test_loader_long):
            if i == BREAKPOINT:
                print(f"  Break after {BREAKPOINT} images.")
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
    print(f"  Finished long dataset {epoch_number} in {epoch_time} seconds!")
