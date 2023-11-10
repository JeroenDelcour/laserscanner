from datetime import datetime
from pathlib import Path

import torch
from torch import nn
import tplot
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm import tqdm, trange

from dataloader import SuperviselyLaserLineDataset
from hourglassnet import HourglassNet

torch.manual_seed(0)

input_size = (512, 512)
output_size = (128, 128)
batch_size = 4
epochs = 600
num_workers = 4

data_transforms = v2.Compose(
    [
        v2.ToImage(),
        v2.Resize(size=input_size, antialias=True),
        v2.ColorJitter(brightness=0.2, hue=0.1),
        # v2.RandomAffine(degrees=45, translate=(0.1, 0.3), scale=(0.5, 1.5)),
        # v2.RandomHorizontalFlip(),
        # v2.RandomVerticalFlip(),
        v2.ToDtype(torch.float32, scale=True),
    ]
)


checkpoint = Path("checkpoints") / (datetime.now().isoformat() + ".pt")
checkpoint.parent.mkdir(parents=True, exist_ok=True)
print(f"Saving checkpoint to: {checkpoint}")

dataset = SuperviselyLaserLineDataset(
    "data/272599_Laser line detection",
    transforms=data_transforms,
    target_shape=output_size,
)
dataloader = DataLoader(
    dataset,
    shuffle=True,
    batch_size=batch_size,
    num_workers=num_workers,
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def criterion(outputs, targets):
    """
    Calculate MSE loss for each hourglass stack and return the sum.
    """
    loss = nn.functional.mse_loss(outputs[0], targets)
    for output in outputs[1:]:
        loss += nn.functional.mse_loss(output, targets)
    return loss


model = HourglassNet(nstack=2, inp_dim=256, oup_dim=1)
model.to(device)

optimizer = Adam(model.parameters(), lr=1e-4)

losses = []

for _ in trange(epochs, unit="epoch"):
    model.train()
    running_loss = 0
    for images, targets in tqdm(dataloader, desc="Training", unit="batch"):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * len(images)
    losses.append(running_loss / len(dataset))

    torch.save(model.state_dict(), checkpoint)

    fig = tplot.Figure(xlabel="epochs", ylabel="loss")
    fig.line(losses)
    fig.show()
