"""
Train a ResNet18 on the poisoned CIFAR-10 training set (BadNets-style,
all2one, target label 2, 5% poison rate) — end-to-end, standard
supervised training. This produces the theta*_poi checkpoint that
ASSET's model_hat step assumes as input.

Run this BEFORE the ASSET_demo.ipynb detection pipeline, then point
`checkpoint_path` in the notebook's cell 5 at the .pth this script saves.
"""
import os
import sys
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from new_poi_util import LocalZipCIFAR10Dataset, my_subset, poi_dataset, set_seed
from models import ResNet18  # same import your notebook uses

# ---- config: keep these identical to what ASSET_demo.ipynb uses ----
ZIP_PATH = "/home/HDD/ATAF/Datasets/CIFAR10-Dataset/CIFAR-10-60k-targeted.zip"          # <-- point at your local CIFAR-10 zip
TAR_LAB = 2
POI_RATES = 0.05
RANDOM_SEED = 0
EPOCHS = 50
BATCH_SIZE = 128
LR = 0.1

# --- pick any poisoning method to test here ---
# 'backdoor'                 # visible patch trigger, all2one (target label fixed)
# 'backdoor_all2all'         # visible patch trigger, all2all (label = original+1 mod num_classes)
# 'noisy_label'              # no trigger, random label noise per class
# 'flipping_label'           # no trigger, flips one source class's labels to one target class
# 'clean_label_narcissus'    # clean-label attack, noise added to target-class images only
# 'noisy_all2one'            # full-image noise trigger, all2one
# 'noisy_all2all'            # full-image noise trigger, all2all
#
# Set via env var so a shell loop can drive this across all methods without editing the file,
# e.g. `POI_METHOD=noisy_label python train_backdoored_model.py`. Defaults to 'backdoor'.
POI_METHOD = os.environ.get('POI_METHOD', 'backdoor_all2all')

FLIP_SRC_LAB = 0   # only used when POI_METHOD == 'flipping_label'

SAVE_PATH = f"/home/HDD/ATAF/siddharth/ASSET/trained_models/poisoned_models/resnet18_cifar10_{POI_METHOD}.pth"

# ---- log everything printed below to logs/train_<POI_METHOD>.log, same as train_all_poisons.sh did via tee ----
class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()

os.makedirs("logs", exist_ok=True)
log_file = open(f"logs/train_{POI_METHOD}.log", "w")
sys.stdout = _Tee(sys.stdout, log_file)
sys.stderr = _Tee(sys.stderr, log_file)

set_seed(RANDOM_SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
test_transform = transforms.Compose([transforms.ToTensor()])

# flipping_label expects tar_lab as (src, dst); every other method expects a single int
tar_lab_arg = (FLIP_SRC_LAB, TAR_LAB) if POI_METHOD == 'flipping_label' else TAR_LAB

# ---- data: same poisoning call as the notebook ----
train_dataset = LocalZipCIFAR10Dataset(ZIP_PATH, train=True)
test_dataset = LocalZipCIFAR10Dataset(ZIP_PATH, train=False)

train_poi_set, o_poi_idx = poi_dataset(
    train_dataset,
    poi_method= POI_METHOD,
    poi_rates=POI_RATES,
    tar_lab=tar_lab_arg,
    transform=train_transform,
    random_seed=RANDOM_SEED,
)

# clean test set for measuring clean accuracy, and a poisoned test set for ASR
clean_test_set = my_subset(test_dataset, list(range(len(test_dataset))), transform=test_transform)
poi_test_set, poi_test_idx = poi_dataset(
    test_dataset,
    poi_method= POI_METHOD,
    poi_rates=1.0,          # trigger every non-target-class test image, for a clean ASR read
    tar_lab=tar_lab_arg,
    transform=test_transform,
    random_seed=RANDOM_SEED,
)

train_loader = DataLoader(train_poi_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
clean_test_loader = DataLoader(clean_test_set, batch_size=512, shuffle=False, num_workers=4)
poi_test_loader = DataLoader(poi_test_set, batch_size=512, shuffle=False, num_workers=4)

model = ResNet18().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)


def evaluate(loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            data, target = batch[0], batch[1]
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            pred = outputs.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    return 100.0 * correct / total


for epoch in range(EPOCHS):
    print(f"Epoch :{epoch}")
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for batch in train_loader:
        data, target = batch[0].to(device), batch[1].to(device)

        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * data.size(0)
        correct += (outputs.argmax(dim=1) == target).sum().item()
        total += target.size(0)

    scheduler.step()
    train_acc = 100.0 * correct / total
    print(f"Epoch {epoch+1}/{EPOCHS} | train_loss={running_loss/total:.4f} | train_acc={train_acc:.2f}%")

    if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
        clean_acc = evaluate(clean_test_loader)
        asr = evaluate(poi_test_loader)
        print(f"  --> Clean ACC: {clean_acc:.2f}% | ASR: {asr:.2f}%")

os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
torch.save(model.state_dict(), SAVE_PATH)
print(f"Saved backdoored checkpoint to {SAVE_PATH}")

final_clean_acc = evaluate(clean_test_loader)
final_asr = evaluate(poi_test_loader)
print(f"Final: Clean ACC={final_clean_acc:.2f}%, ASR={final_asr:.2f}%")
