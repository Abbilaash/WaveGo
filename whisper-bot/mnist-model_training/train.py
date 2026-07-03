# CNN with MNIST dataset

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision.datasets import EMNIST
from torchvision import transforms

from torch.utils.data import DataLoader

# =========================
# CONFIG
# =========================

BATCH_SIZE = 128
EPOCHS = 50
LR = 0.001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)

# =========================
# DATASET
# =========================

transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = EMNIST(
    root="data",
    split="digits",
    train=True,
    download=True,
    transform=transform
)

test_dataset = EMNIST(
    root="data",
    split="digits",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# =========================
# MODEL
# =========================

class DigitCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        self.pool = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(
            64 * 7 * 7,
            128
        )

        self.fc2 = nn.Linear(
            128,
            10
        )

    def forward(self, x):

        x = torch.relu(self.conv1(x))
        x = self.pool(x)

        x = torch.relu(self.conv2(x))
        x = self.pool(x)

        x = x.view(x.size(0), -1)

        x = torch.relu(self.fc1(x))

        x = self.fc2(x)

        return x

model = DigitCNN().to(DEVICE)

# =========================
# LOSS + OPTIMIZER
# =========================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LR
)

# =========================
# TRAIN
# =========================

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)

    print(
        f"Epoch {epoch+1}/{EPOCHS} "
        f"Loss = {avg_loss:.4f}"
    )

# =========================
# TEST
# =========================

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        preds = outputs.argmax(dim=1)

        correct += (
            preds == labels
        ).sum().item()

        total += labels.size(0)

accuracy = 100 * correct / total

print(
    f"\nTest Accuracy: {accuracy:.2f}%"
)

# =========================
# SAVE PYTORCH MODEL
# =========================

torch.save(
    model.state_dict(),
    "digit_cnn.pth"
)

print(
    "Saved digit_cnn.pth"
)

# =========================
# EXPORT ONNX
# =========================

dummy_input = torch.randn(
    1,
    1,
    28,
    28
).to(DEVICE)

torch.onnx.export(
    model,
    dummy_input,
    "digit_cnn.onnx",

    input_names=["input"],
    output_names=["output"],

    opset_version=11,
    export_params=True,

    do_constant_folding=True
)

print(
    "Saved digit_cnn.onnx"
)
