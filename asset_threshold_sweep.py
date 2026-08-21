"""
Sweeps the ASSET active-separation training threshold (adjusted_outlyingness cutoff)
from 1.0 to 2.0 in steps of 0.1, across all 14 poisoning methods. Resume logic skips
any (method, threshold) pair already logged in RESULTS_CSV, so methods already swept
in an earlier run (backdoor_all2all, noisy_label, flipping_label,
clean_label_narcissus, noisy_all2all) are left untouched, and the 7 newly-integrated
ATAF methods (wanet, blended, badnets, low_frequency, ssba, inputaware, ctrl) run
fresh. 'backdoor' and 'noisy_all2one' are pre-seeded with dummy all-zero rows for
every threshold (best thresholds already found manually elsewhere: 1.1 and 1.3
respectively) purely so this script skips them instead of re-sweeping from scratch --
those dummy rows are not real results and must not be read as such.

For each poisoning method, only the ONE threshold that gives the best AUC has its
trained model kept -- saved to ASSET_BEST_MODELS_DIR, overwriting the previous best
for that method whenever a better one is found. Every attempt (not just the best) is
logged to RESULTS_CSV so the full sweep is reviewable afterward.

Run this in the background (e.g. via tmux) and check back later:
    tmux new-session -d -s threshold_sweep "cd /home/siddarth/ASSET && source /home/siddarth/UV-ENV/bin/activate && python asset_threshold_sweep.py"
"""
import os
import sys
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose
from sklearn.metrics import roc_auc_score

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from new_poi_util import (LocalZipCIFAR10Dataset, my_subset, poi_dataset, set_seed,
                           get_result, get_t, adjusted_outlyingness)
from models import ResNet18

# ---- config ----
ZIP_PATH = "/home/HDD/ATAF/Datasets/CIFAR10-Dataset/CIFAR-10-60k-targeted.zip"
TAR_LAB = 2
FLIP_SRC_LAB = 0
MODEL_NAME = "resnet18_cifar10"
EPSILON = 1e-50
BATCH_SIZE = 512
NWORKERS = 8
VALID_SIZE = 1000
EPOCHS = 2
RANDOM_SEED = 0

POI_METHODS = ["backdoor", "backdoor_all2all", "noisy_label", "flipping_label",
               "clean_label_narcissus", "noisy_all2one", "noisy_all2all",
               "badnets", "inputaware", "ctrl",
               "wanet", "blended", "low_frequency", "ssba",
               ]
THRESHOLDS = [round(1.0 + 0.1 * i, 1) for i in range(11)]  # 1.0, 1.1, ..., 2.0

BEST_MODELS_DIR = "/home/HDD/ATAF/siddharth/ASSET/trained_models/asset_best_models"
RESULTS_CSV = "results_csv/asset_threshold_sweep_results.csv"


# ---- log everything printed below to logs/asset_threshold_sweep.log ----
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
os.makedirs(BEST_MODELS_DIR, exist_ok=True)
log_file = open("logs/asset_threshold_sweep.log", "w")
sys.stdout = _Tee(sys.stdout, log_file)
sys.stderr = _Tee(sys.stderr, log_file)

set_seed(RANDOM_SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
transform = Compose([ToTensor()])


class HiddenLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc(x))


class MLP(nn.Module):
    def __init__(self, input_size=10, hidden_size=100, num_layers=1):
        super().__init__()
        self.first_hidden_layer = HiddenLayer(input_size, hidden_size)
        self.rest_hidden_layers = nn.Sequential(*[HiddenLayer(hidden_size, hidden_size) for _ in range(num_layers - 1)])
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.first_hidden_layer(x)
        x = self.rest_hidden_layers(x)
        x = self.output_layer(x)
        return torch.sigmoid(x)


def log_row(rows_buffer, poi_method, threshold, auc, tp, fp, fn, clean_total):
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr_rate = fp / clean_total if clean_total > 0 else 0.0
    rows_buffer.append({
        "POI_METHOD": poi_method, "Threshold": threshold, "Epsilon": EPSILON, "AUC": auc,
        "TP": tp, "FP": fp, "FN": fn, "Recall/TPR": recall, "Precision": precision,
        "F1": f1, "FPR": fpr_rate,
    })
    df = pd.DataFrame(rows_buffer)
    df.to_csv(RESULTS_CSV, index=False)


# ---- shared clean validation pool (built once, same for every method/threshold) ----
val_dataset = LocalZipCIFAR10Dataset(ZIP_PATH, train=False, transform=None)
val_idx = []
for i in range(10):
    current_label = np.where(np.array(val_dataset.targets) == i)[0]
    samples_idx = np.random.choice(current_label, size=int(VALID_SIZE / 10), replace=False)
    val_idx.extend(samples_idx)
val_set = my_subset(val_dataset, val_idx, transform=transform)

train_dataset = LocalZipCIFAR10Dataset(ZIP_PATH, train=True, transform=None)

rows_buffer = []
completed = set()  # {(POI_METHOD, THRESHOLD)} already logged from a prior (possibly killed) run
if os.path.exists(RESULTS_CSV):
    rows_buffer = pd.read_csv(RESULTS_CSV).to_dict("records")
    completed = {(r["POI_METHOD"], round(float(r["Threshold"]), 1)) for r in rows_buffer}
    print(f"Resuming: found {len(rows_buffer)} previously logged run(s) in {RESULTS_CSV}")

for POI_METHOD in POI_METHODS:
    remaining_thresholds = [t for t in THRESHOLDS if (POI_METHOD, t) not in completed]
    if not remaining_thresholds:
        print(f"\n=== POI_METHOD = {POI_METHOD}: all thresholds already completed, skipping ===")
        continue

    print(f"\n{'='*70}\n=== POI_METHOD = {POI_METHOD} ===\n{'='*70}")
    if len(remaining_thresholds) < len(THRESHOLDS):
        done_already = [t for t in THRESHOLDS if (POI_METHOD, t) in completed]
        print(f"  Resuming this method: {len(done_already)} threshold(s) already done {done_already}, "
              f"{len(remaining_thresholds)} remaining {remaining_thresholds}")

    tar_lab_arg = (FLIP_SRC_LAB, TAR_LAB) if POI_METHOD == "flipping_label" else TAR_LAB

    # ---- poisoning + o_model setup: same for every threshold at this method ----
    train_poi_set, o_poi_idx = poi_dataset(
        train_dataset, poi_method=POI_METHOD, transform=transform,
        poi_rates=0.05, random_seed=RANDOM_SEED, tar_lab=tar_lab_arg,
    )
    train_dataloader = DataLoader(train_poi_set, batch_size=BATCH_SIZE, num_workers=NWORKERS,
                                   pin_memory=True, shuffle=True)

    checkpoint_path = f"/home/HDD/ATAF/siddharth/ASSET/trained_models/poisoned_models/resnet18_cifar10_{POI_METHOD}.pth"
    if not os.path.exists(checkpoint_path):
        print(f"  !! No poisoned checkpoint found at {checkpoint_path}, skipping {POI_METHOD}")
        continue

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    o_model = ResNet18()
    o_model.load_state_dict(state)
    o_model = o_model.to(device)
    o_model.eval()

    o_model2 = copy.deepcopy(o_model)
    o_model2.linear = nn.Identity()
    model_hat = o_model2.to(device)
    model_hat.eval()
    for p in model_hat.parameters():
        p.requires_grad = False

    best_model_path = f"{BEST_MODELS_DIR}/asset_{MODEL_NAME}_{POI_METHOD}_best.pth"

    # re-derive this method's running-best AUC from already-completed rows (if resuming),
    # so a restart keeps comparing new attempts against true history, not starting fresh at -1
    prior_rows = [r for r in rows_buffer if r["POI_METHOD"] == POI_METHOD]
    if prior_rows:
        best_row = max(prior_rows, key=lambda r: r["AUC"])
        best_auc = best_row["AUC"]
        best_threshold = round(float(best_row["Threshold"]), 1)
        print(f"  Prior best for {POI_METHOD}: threshold={best_threshold}, AUC={best_auc:.4f} "
              f"(best model file should already exist on disk from before)")
    else:
        best_auc = -1.0
        best_threshold = None

    for THRESHOLD in remaining_thresholds:
        print(f"\n--- {POI_METHOD} | threshold={THRESHOLD} ---")

        model = ResNet18().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        optimizer2 = torch.optim.Adam(model.parameters(), lr=0.0001)
        criterion = nn.CrossEntropyLoss()
        mse = nn.MSELoss()

        for epoch in range(EPOCHS):
            model.train()
            for iters, batch in enumerate(train_dataloader):
                pos_img, pos_lab = batch[0].to(device), batch[1].to(device)

                idxs = random.sample(range(VALID_SIZE), min(BATCH_SIZE, VALID_SIZE))
                neg_img = torch.stack([val_set[i][0] for i in idxs]).to(device)
                neg_lab = torch.tensor([val_set[i][1] for i in idxs]).to(device)

                neg_outputs = model(neg_img)
                neg_loss = torch.mean(torch.var(neg_outputs, dim=1))
                optimizer.zero_grad()
                neg_loss.backward()
                optimizer.step()

                # model_hat is frozen and pos_img/neg_img are fixed for this whole inner loop,
                # so compute its forward pass once and reuse instead of recomputing it 100 times.
                with torch.no_grad():
                    v_outputs = model_hat(pos_img)
                    vn_outputs = model_hat(neg_img)
                feature_dim = v_outputs.shape[1]

                Vnet = MLP(input_size=feature_dim, hidden_size=128, num_layers=2).to(device)
                Vnet.train()
                optimizer_hat = torch.optim.Adam(Vnet.parameters(), lr=0.0001)
                optimizer_hat2 = torch.optim.Adam(Vnet.parameters(), lr=0.0001)

                for _ in range(100):
                    vneto = Vnet(v_outputs)
                    v_label = torch.ones(v_outputs.shape[0]).to(device)
                    rr_loss = mse(vneto.view(-1), v_label)
                    Vnet.zero_grad()
                    rr_loss.backward()
                    optimizer_hat.step()

                    vneto2 = Vnet(vn_outputs)
                    v_label2 = torch.zeros(vn_outputs.shape[0]).to(device)
                    rr_loss2 = mse(vneto2.view(-1), v_label2)
                    Vnet.zero_grad()
                    rr_loss2.backward()
                    optimizer_hat2.step()

                with torch.no_grad():
                    res = Vnet(v_outputs)

                pidx = torch.where(adjusted_outlyingness(res) > THRESHOLD)[0]
                if pidx.numel() == 0:
                    continue

                pos_outputs = model(pos_img[pidx])
                real_loss = -criterion(pos_outputs, pos_lab[pidx])
                optimizer2.zero_grad()
                real_loss.backward()
                optimizer2.step()

            print(f" epoch {epoch+1}/{EPOCHS} done")

        poi_res, clean_res = get_result(model, train_poi_set, o_poi_idx)
        true_label = [1] * len(poi_res) + [0] * len(clean_res)
        pred_label = poi_res + clean_res
        auc = roc_auc_score(true_label, pred_label)

        total = poi_res + clean_res
        t = get_t(total, EPSILON)
        tp = len(o_poi_idx) - np.where(np.array(poi_res) < t)[0].shape[0]
        fp = len(clean_res) - np.where(np.array(clean_res) < t)[0].shape[0]
        fn = np.where(np.array(poi_res) < t)[0].shape[0]

        print(f"  AUC={auc:.4f}  tp={tp}  fp={fp}  fn={fn}")
        log_row(rows_buffer, POI_METHOD, THRESHOLD, auc, tp, fp, fn, len(clean_res))

        if auc > best_auc:
            best_auc = auc
            best_threshold = THRESHOLD
            torch.save(model.state_dict(), best_model_path)
            print(f"  ** new best for {POI_METHOD}: threshold={THRESHOLD}, AUC={auc:.4f} -> saved to {best_model_path}")

    print(f"\n>>> {POI_METHOD} finished. Best threshold={best_threshold}, AUC={best_auc:.4f}, saved at {best_model_path}")

print("\n=== SWEEP COMPLETE ===")
print(pd.DataFrame(rows_buffer).to_string(index=False))
