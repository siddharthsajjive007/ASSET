#!/usr/bin/env python3
"""
Retrains only the 4 poisoning methods that came out with weak ASR
(blended, wanet, low_frequency, ssba), trying a few epoch/poison-rate
combinations chosen per what we diagnosed for each method:
  - wanet: ASR was still climbing at epoch 50 -> try more epochs
  - blended: moderate ASR -> more epochs + a bit more poison rate
  - low_frequency: subtle but consistent trigger -> poison rate helps more than epochs
  - ssba: trigger is sample-specific (no shared pattern across poisoned images) --
    unlikely to improve much via epochs/poison-rate alone, included anyway

Only OVERWRITES the saved checkpoint for a method when a new attempt's final ASR
beats the best ASR recorded so far for that method (seeded below from the
existing logs, then tracked persistently in RESULTS_CSV so re-running this later
with more attempts keeps comparing against true history, not restarting at zero).
Every attempt (not just the winning ones) is logged to RESULTS_CSV.

Run in the background, e.g.:
    tmux new-session -d -s retrain_weak "/home/siddarth/ASSET/retrain_weak_poisons.py"
(make sure it's executable: chmod +x retrain_weak_poisons.py, and has a shebang --
 or just run: python retrain_weak_poisons.py)
"""
import os
import sys
import copy
import csv
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from new_poi_util import LocalZipCIFAR10Dataset, my_subset, poi_dataset, set_seed
from models import ResNet18

# ---- config ----
ZIP_PATH = "/home/HDD/ATAF/Datasets/CIFAR10-Dataset/CIFAR-10-60k-targeted.zip"
TAR_LAB = 2
FLIP_SRC_LAB = 0
RANDOM_SEED = 0
BATCH_SIZE = 128
LR = 0.1

CHECKPOINT_DIR = "/home/HDD/ATAF/siddharth/ASSET/trained_models/poisoned_models"
RESULTS_CSV = "results_csv/retrain_weak_poisons_results.csv"
OTHER_RESULTS_CSV = "results_csv/retrain_one_results.csv"  # read-only, owned by retrain_one_poison.py

# Baseline ASR already achieved (from logs/train_<method>.log) -- new attempts
# must beat this to overwrite the saved checkpoint.
BASELINE_ASR = {
    "wanet": 35.06,
    "blended": 88.7,
    "low_frequency": 91.54,
    "ssba": 12.57,
    "flipping_label": 85.08,
    "noisy_all2all": 89.82,
}

# (method, epochs, poi_rates) attempts to try.
ATTEMPTS = [("blended", 50, 0.15),
]

# ---- log everything to logs/retrain_weak_poisons.log ----
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
log_file = open("logs/retrain_weak_poisons.log", "w")
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

# ---- raw datasets loaded once, reused for every attempt ----
train_dataset = LocalZipCIFAR10Dataset(ZIP_PATH, train=True)
test_dataset = LocalZipCIFAR10Dataset(ZIP_PATH, train=False)


def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            data, target = batch[0].to(device), batch[1].to(device)
            outputs = model(data)
            pred = outputs.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    return 100.0 * correct / total


def run_attempt(poi_method, epochs, poi_rates):
    tar_lab_arg = (FLIP_SRC_LAB, TAR_LAB) if poi_method == "flipping_label" else TAR_LAB

    train_poi_set, _ = poi_dataset(
        train_dataset, poi_method=poi_method, poi_rates=poi_rates,
        tar_lab=tar_lab_arg, transform=train_transform, random_seed=RANDOM_SEED,
    )
    clean_test_set = my_subset(test_dataset, list(range(len(test_dataset))), transform=test_transform)
    poi_test_set, _ = poi_dataset(
        test_dataset, poi_method=poi_method, poi_rates=1.0,
        tar_lab=tar_lab_arg, transform=test_transform, random_seed=RANDOM_SEED,
    )

    train_loader = DataLoader(train_poi_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    clean_test_loader = DataLoader(clean_test_set, batch_size=512, shuffle=False, num_workers=4)
    poi_test_loader = DataLoader(poi_test_set, batch_size=512, shuffle=False, num_workers=4)

    model = ResNet18().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
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
        print(f"  epoch {epoch+1}/{epochs} | train_loss={running_loss/total:.4f} | train_acc={train_acc:.2f}%")

        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            clean_acc = evaluate(model, clean_test_loader)
            asr = evaluate(model, poi_test_loader)
            print(f"    --> Clean ACC: {clean_acc:.2f}% | ASR: {asr:.2f}%")

    final_clean_acc = evaluate(model, clean_test_loader)
    final_asr = evaluate(model, poi_test_loader)
    return model, final_clean_acc, final_asr


CSV_FIELDS = ["method", "epochs", "poi_rates", "final_clean_acc", "final_asr"]


def load_best_asr():
    """Re-read both RESULTS_CSV and OTHER_RESULTS_CSV fresh each time -- safe to
    call even while another process (e.g. retrain_one_poison.py) is concurrently
    appending to either file, since this never writes to either."""
    best = dict(BASELINE_ASR)
    for path in (RESULTS_CSV, OTHER_RESULTS_CSV):
        if os.path.exists(path):
            for r in pd.read_csv(path).to_dict("records"):
                m, asr = r["method"], r["final_asr"]
                if asr > best.get(m, -1.0):
                    best[m] = asr
    return best


def append_row(row):
    """True single-row append (O_APPEND), never reads or rewrites existing
    content -- safe for multiple processes to append to the same RESULTS_CSV
    concurrently, unlike read-all/rewrite-all."""
    write_header = not os.path.exists(RESULTS_CSV) or os.path.getsize(RESULTS_CSV) == 0
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ---- resume support: pick up best-so-far from a prior run of this script, if any ----
best_asr = load_best_asr()
if os.path.exists(RESULTS_CSV):
    print(f"Resuming: found {len(pd.read_csv(RESULTS_CSV))} previous attempt(s) in {RESULTS_CSV}")

print(f"Baseline ASR to beat: {best_asr}")

for poi_method, epochs, poi_rates in ATTEMPTS:
    best_asr = load_best_asr()  # refresh in case another process appended meanwhile
    current_best = best_asr.get(poi_method, -1.0)
    print(f"\n{'='*70}\n=== {poi_method} | epochs={epochs} poi_rates={poi_rates} "
          f"(current best ASR={current_best:.2f}%) ===\n{'='*70}")

    model, clean_acc, asr = run_attempt(poi_method, epochs, poi_rates)

    # refresh right before the overwrite decision (before appending our own row below,
    # or load_best_asr() would see this attempt's own row and always compare asr to itself),
    # in case another process improved this same method while we were training
    current_best = max(current_best, load_best_asr().get(poi_method, -1.0))

    row = {
        "method": poi_method, "epochs": epochs, "poi_rates": poi_rates,
        "final_clean_acc": clean_acc, "final_asr": asr,
    }
    append_row(row)

    if asr > current_best:
        ckpt_path = f"{CHECKPOINT_DIR}/resnet18_cifar10_{poi_method}.pth"
        torch.save(model.state_dict(), ckpt_path)
        best_asr[poi_method] = asr
        print(f"  ** NEW BEST for {poi_method}: ASR {asr:.2f}% (was {current_best:.2f}%) "
              f"-- saved to {ckpt_path}")
    else:
        print(f"  Not an improvement for {poi_method}: ASR {asr:.2f}% <= current best {current_best:.2f}% "
              f"-- checkpoint left unchanged")

print("\n=== ALL ATTEMPTS COMPLETE ===")
print(f"Final best ASR per method: {best_asr}")
print(pd.read_csv(RESULTS_CSV).to_string(index=False))
