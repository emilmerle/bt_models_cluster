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
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

training_set = torchvision.datasets.CIFAR10(
    root="./../../data", train=True, download=True, transform=transform
)

VALIDATION_BATCHSIZE = 1
validation_set = torchvision.datasets.CIFAR10(
    root="./../../data", train=False, download=True, transform=transform
)
validation_loader = torch.utils.data.DataLoader(
    validation_set, batch_size=VALIDATION_BATCHSIZE, shuffle=True
)

# Report split sizes
len_training_set = len(training_set)
len_validation_set = len(validation_set)
print(f"Training set has {len_training_set} instances")
print(f"Validation set has {len_validation_set} instances")
print()

classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# CUDA or CPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using {str(device).upper()}!\n")

model = Net()
model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


filename = f"./../../output_data/Training_Batchsizes_CNN_CIFAR10/{device_name}.csv"
os.makedirs(os.path.dirname(filename), exist_ok=True)
model_path = "./../../models/"
os.makedirs(os.path.dirname(model_path), exist_ok=True)

EPOCHS = 7
if device_name in ["RP3", "RP4", "JETSONNANO"]:
    BREAKPOINT_TRAINING = 5000
    BREAKPOINT_VALIDATION = 1000
else:
    BREAKPOINT_TRAINING = 5000
    BREAKPOINT_VALIDATION = 1000

best_vloss = 1_000_000.0

BATCHSIZES = [1, 4, 16, 64, 256]

full_start_time = time.time()
for batchsize in BATCHSIZES:
    CURRENT_TRAINING_BATCHSIZE = batchsize
    CURRENT_BREAKPOINT = (BREAKPOINT_TRAINING // CURRENT_TRAINING_BATCHSIZE) + 1
    training_loader = torch.utils.data.DataLoader(
        training_set, batch_size=CURRENT_TRAINING_BATCHSIZE, shuffle=True
    )
    for epoch in range(EPOCHS):  # loop over the dataset multiple times
        # TRAINING
        print(
            f"Starting Epoch {epoch + 1} with Batchsize {CURRENT_TRAINING_BATCHSIZE}:"
        )
        epoch_train_start_time = time.time()

        model.train(True)

        # Train one Epoch:
        running_loss = 0.0
        last_loss = 0.0

        batches_done = 0
        for i, data in enumerate(training_loader):
            if i == CURRENT_BREAKPOINT:
                print(
                    f"  Break training after {CURRENT_BREAKPOINT} batches (batchsize {CURRENT_TRAINING_BATCHSIZE})."
                )
                break

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 1000 == 999:
                last_loss = running_loss / 1000  # loss per batch
                running_loss = 0.0

            batches_done += 1

        avg_loss = last_loss

        epoch_train_end_time = time.time()
        epoch_train_time = epoch_train_end_time - epoch_train_start_time

        # EVALUATION
        epoch_eval_start_time = time.time()
        running_vloss = 0.0
        model.eval()

        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                if i == BREAKPOINT_VALIDATION:
                    print(
                        f"  Break validation after {BREAKPOINT_VALIDATION} Batches (Validation-Batchsize {VALIDATION_BATCHSIZE})."
                    )
                    break
                vinputs, vlabels = vdata[0].to(device), vdata[1].to(device)
                voutputs = model(vinputs)
                vloss = criterion(voutputs, vlabels)
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)

        epoch_eval_end_time = time.time()
        epoch_eval_time = epoch_eval_end_time - epoch_eval_start_time

        # REPORTING
        print(f"  Loss for epoch {epoch + 1}:")
        print(f"    Train:      {avg_loss}")
        print(f"    Validation: {avg_vloss}")
        print()

        images_processed = min(
            batches_done * CURRENT_TRAINING_BATCHSIZE, len_training_set
        )
        row = [
            device_name,
            images_processed,
            epoch_train_time,
            avg_loss,
            CURRENT_TRAINING_BATCHSIZE,
            BREAKPOINT_VALIDATION,
            epoch_eval_time,
            avg_vloss.item(),
            VALIDATION_BATCHSIZE,
        ]

        with open(filename, "a", newline="") as csvfile:
            csvwriter = csv.writer(
                csvfile, delimiter=",", quoting=csv.QUOTE_NONE, escapechar="\\"
            )
            csvwriter.writerow(row)

        # SAVING
        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = f"./../../models/CNN_CIFAR10_trained_model"
            torch.save(model.state_dict(), model_path)


full_end_time = time.time()
full_training_time = full_end_time - full_start_time
print(f"Finished Training in {full_training_time} seconds!")
