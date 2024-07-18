import torch
import torchvision
import torchvision.transforms as transforms

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

training_set = torchvision.datasets.Caltech256(
    "./../../data/caltech256", transform=transform, download=True
)

VALIDATION_BATCHSIZE = 1
validation_set = torchvision.datasets.Caltech256(
    "./../../data/caltech256", transform=transform, download=True
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


loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


filename = (
    f"./../../output_data/Training_Batchsizes_Resnet50_Caltech256/{device_name}.csv"
)
os.makedirs(os.path.dirname(filename), exist_ok=True)

EPOCHS = 10
if device_name in ["RP3", "RP4", "JETSONNANO"]:
    BREAKPOINT_TRAINING = 100
    BREAKPOINT_VALIDATION = 100
elif device_name == "CLUSTER":
    BREAKPOINT_TRAINING = 1_000_000
    BREAKPOINT_VALIDATION = 1_000_000
else:
    BREAKPOINT_TRAINING = 512
    BREAKPOINT_VALIDATION = 100

best_vloss = 1_000_000.0

BATCHSIZES = [1, 4, 16, 64, 256]

full_start_time = time.time()

for batchsize in BATCHSIZES:
    CURRENT_TRAINING_BATCHSIZE = batchsize
    CURRENT_BREAKPOINT = (BREAKPOINT_TRAINING // CURRENT_TRAINING_BATCHSIZE) + 1
    training_loader = torch.utils.data.DataLoader(
        training_set, batch_size=CURRENT_TRAINING_BATCHSIZE, shuffle=True
    )
    for epoch in range(EPOCHS):
        # TRAINING
        print(
            f"Starting Epoch {epoch + 1} with Batchsize {CURRENT_TRAINING_BATCHSIZE}:"
        )
        epoch_train_start_time = time.time()

        model.train()

        # Train one Epoch
        running_loss = 0.0
        last_loss = 0.0

        batches_done = 0

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in enumerate(training_loader):
            # BREAK TO SPEED UP TRAINING
            if i == CURRENT_BREAKPOINT:
                print(
                    f"  Break training after {CURRENT_BREAKPOINT} batches (batchsize {CURRENT_TRAINING_BATCHSIZE})."
                )
                break
            # Every data instance is an input + label pair
            inputs, labels = data[0].to(device), data[1].to(device)

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            outputs = model(inputs)

            # Compute the loss and its gradients
            loss = loss_fn(outputs, labels)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
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
                vloss = loss_fn(voutputs, vlabels)
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
            model_path = f"./../../models/Resnet50_Caltech256_trained_model"
            torch.save(model.state_dict(), model_path)


full_end_time = time.time()
full_training_time = full_end_time - full_start_time
print(f"Finished Training in {full_training_time} seconds!")
