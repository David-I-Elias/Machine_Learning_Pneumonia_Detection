import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import copy
import time
import matplotlib.pyplot as plt
import numpy as np
import os

EPOCHS = 1
data_directory = "chest_xray\chest_xray"
TEST = 'test'
TRAIN = 'train'
VAL = 'val'


def data_transforms(phase):
    if phase == TRAIN:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    if phase == VAL:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    if phase == TEST:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    return transform


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("Cuda supported device detected, using GPU\n")
else:
    print("Using CPU, see you in 20 years!\n")

xray_datasets = {x: datasets.ImageFolder(os.path.join(data_directory, x), data_transforms(x)) for x in
                 [TRAIN, VAL, TEST]}

dataloaders = {TRAIN: torch.utils.data.DataLoader(xray_datasets[TRAIN], batch_size=16, shuffle=True),
               VAL: torch.utils.data.DataLoader(xray_datasets[VAL], batch_size=1, shuffle=True),
               TEST: torch.utils.data.DataLoader(xray_datasets[TEST], batch_size=1, shuffle=True)}

dataset_sizes = {x: len(xray_datasets[x]) for x in [TRAIN, VAL]}
classes = xray_datasets[TRAIN].classes
class_names = xray_datasets[TRAIN].classes


def xrayVisualization(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.show()


inputs, classes = next(iter(dataloaders[TRAIN]))
out = torchvision.utils.make_grid(inputs)
xrayVisualization(out, title=[class_names[x] for x in classes])

inputs, classes = next(iter(dataloaders[TRAIN]))


def train_model(model, loss_function, optimizer, scheduler, num_epochs):
    best_accuracy = 0.0
    best_model = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        print("Epoch: ", epoch + 1, "/", num_epochs)
        print("=" * 20)

        for phase in [TRAIN, VAL]:
            if phase == TRAIN:
                scheduler.step()
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            for data in dataloaders[phase]:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == TRAIN):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = loss_function(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Accuracy: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_accuracy:
                best_accuracy = epoch_acc
                best_model = copy.deepcopy(model.state_dict())

    print('Best val Accuracy: {:4f}'.format(best_accuracy))
    model.load_state_dict(best_model)
    return model


def test_model():
    testing_correct = 0
    testing_total = 0
    true_labels = []
    prediction_labels = []

    with torch.no_grad():
        for data in dataloaders[TEST]:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            true_labels.append(labels.item())
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            prediction_labels.append(preds.item())
            testing_total += labels.size(0)
            testing_correct += (preds == labels).sum().item()
        acc = testing_correct / testing_total
    return true_labels, prediction_labels, testing_correct, testing_total, acc


model = models.resnet152()

model = model.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.01)
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

training_start_time = time.time()
print("Beginning Training. Please be patient as training can take a long time depending on model and hardware used\n")
model = train_model(model, loss_function, optimizer, exp_lr_scheduler, num_epochs=EPOCHS)
print("/" * 20)

print("Training completed")
training_time_taken = time.time() - training_start_time
print("Training complete in {: .0f} m {: .0f} s\n".format(training_time_taken // 60, training_time_taken % 60))

testing_start_time = time.time()
print("Continuing to testing phase. Please wait.\n")
true_labels, prediction_labels, testing_correct, testing_total, acc = test_model()

print("Testing completed")
testing_time_taken = time.time() - testing_start_time
print("Testing complete in {: .0f} m {: .0f} s".format(testing_time_taken // 60, testing_time_taken % 60))
print("Total Correct: ", testing_correct, "out of: ", testing_total, "total images")
print("Test Accuracy: ", acc)
