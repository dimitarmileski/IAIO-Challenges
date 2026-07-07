import numpy as np
import torch
import torch.nn as nn
import zipfile

import os

torch.manual_seed(42)
np.random.seed(42)

torch.cuda.is_available()

# Load training data from subtask1/
def load_subtask1_data():
    with zipfile.ZipFile("/home/jovyan/work/t1/train_data.zip", "r") as zf:
        train_data = {}
        for name in zf.namelist():
            if name.startswith("subtask1/") and name.endswith(".npy"):
                with zf.open(name) as f:
                    train_data[name] = np.load(f)
    
    # Concatenate all three scanners
    sub1_X = []
    sub1_y = []
    for s in [1, 2, 3]:
        x = train_data[f"subtask1/scanner{s}_X.npy"].astype(np.float32) / 255.0
        y = train_data[f"subtask1/scanner{s}_y.npy"].astype(np.int64)
        sub1_X.append(x)
        sub1_y.append(y)
    X1 = np.concatenate(sub1_X, axis=0)
    y1 = np.concatenate(sub1_y, axis=0)
    print(X1.shape, y1.shape)
    return X1, y1

# Load training data from subtask2/
def load_subtask2_data():
    with zipfile.ZipFile("/home/jovyan/work/t1/train_data.zip", "r") as zf:
        train_data = {}
        for name in zf.namelist():
            if name.startswith("subtask2/") and name.endswith(".npy"):
                with zf.open(name) as f:
                    train_data[name] = np.load(f)

    # Concatenate all eight scanners
    sub2_X = []
    sub2_y = []
    for s in range(1, 9):
        x = train_data[f"subtask2/scanner{s}_X.npy"].astype(np.float32) / 255.0
        y = train_data[f"subtask2/scanner{s}_y.npy"].astype(np.int64)
        sub2_X.append(x)
        sub2_y.append(y)
    X2 = np.concatenate(sub2_X, axis=0)
    y2 = np.concatenate(sub2_y, axis=0)
    print(X2.shape, y2.shape)
    return X2, y2

class Sub1CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 4, 3, padding=1), nn.BatchNorm2d(4), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(4, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(16 * 3 * 3, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        # Handles both 1-ch (subtask1) and 3-ch (subtask2) input.
        # For subtask2, converts RGB to grayscale!
        if x.shape[1] == 3:
            x = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        return self.net(x)

class Sub2CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv2d(3,8,3,padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(32 * 3 * 3, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )
        
    def forward(self,x):
        return self.net(x)

def train_model(model, X, y, epochs=30, lr=0.005, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    X_t = torch.from_numpy(X).to(device)
    y_t = torch.from_numpy(y).long().to(device)

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(y))
        total_loss = 0.0
        for start in range(0, len(y), batch_size):
            idx = perm[start:start + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(X_t[idx]), y_t[idx])
            loss.backward()
            opt.step()
            total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            acc = (model(X_t).argmax(1) == y_t).float().mean().item()
        print(f"  epoch {epoch:2d}: loss={total_loss:.4f}  acc={acc:.4f}")

    model.eval()
    return model.cpu()

X1, y1 = load_subtask1_data()

X2, y2 = load_subtask2_data()

# Train Subtask 1 starter model
model1 = Sub1CNN()
params1 = sum(p.numel() for p in model1.parameters())
print(f"  params: {params1:,}")
model1 = train_model(model1, X1, y1)

# Export Subtask 1 model with jit.script
model1.eval();
script1 = torch.jit.script(model1)
torch.jit.save(script1, "model_sub1.pt")
print(f"Exported model_sub1.pt")

# Train Subtask 2 starter model
model2 = Sub2CNN()
params2 = sum(p.numel() for p in model2.parameters())
print(f"  params: {params2:,}")
model2 = train_model(model2, X2, y2, epochs=60, lr=0.001)

# Export Subtask 2 model with jit.script
model2.eval();
script2 = torch.jit.script(model2)
torch.jit.save(script2, "model_sub2.pt")
print(f"Exported model_sub2.pt")

# Make zip submission
with zipfile.ZipFile("submission.zip", "w") as zf:
    zf.write("model_sub1.pt")
    zf.write("model_sub2.pt")
print("Wrote submission.zip!")


