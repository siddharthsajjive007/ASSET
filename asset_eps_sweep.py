#!/usr/bin/env python3
"""
For every method with a saved ASSET detective-model checkpoint in asset_best_models/,
loads that model ONCE (the same "run once, then just swap eps and re-run the last
cell" workflow you do by hand in the notebook) and exhaustively sweeps EPSILON over
every power of ten from 1 (1e0) down to 1e-50, finding the eps with the best tp/fp/fn
trade-off (best F1) for each method.

Checkpoints are discovered dynamically via glob (each is named
asset_<model>_<poi_method>_<threshold>.pth, with a different threshold value baked
into each filename per method), so a method still being produced by a running
asset_threshold_sweep.py just won't appear yet -- re-run this script later to pick
it up once it's saved.

Read-only against asset_best_models/, one forward pass per method (no training), so
it's safe and cheap to run alongside asset_threshold_sweep.py.

Run:
    python asset_eps_sweep.py
"""
import os
import sys
import glob
import numpy as np
import pandas as pd
import torch
from torchvision.transforms import ToTensor, Compose
from sklearn.metrics import roc_auc_score

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from new_poi_util import LocalZipCIFAR10Dataset, poi_dataset, set_seed, get_result, get_t
from models import ResNet18

ZIP_PATH = "/home/HDD/ATAF/Datasets/CIFAR10-Dataset/CIFAR-10-60k-targeted.zip"
TAR_LAB = 2
FLIP_SRC_LAB = 0
RANDOM_SEED = 0
MODEL_NAME = "resnet18_cifar10"

ASSET_BEST_DIR = "/home/HDD/ATAF/siddharth/ASSET/trained_models/asset_best_models"
RESULTS_CSV = "results_csv/asset_eps_sweep_results.csv"

# all 14 poisoning methods -- checkpoints are matched by prefix
# (asset_<model>_<poi_method>_<threshold>.pth), so the threshold suffix doesn't matter
ALL_POI_METHODS = [
    "backdoor", "backdoor_all2all", "noisy_label", "flipping_label",
    "clean_label_narcissus", "noisy_all2one", "noisy_all2all",
    "wanet", "badnets", "low_frequency", "ssba",
    "inputaware", "ctrl",
    # "blended",
]

EXPONENTS = list(range(0, 101))  # eps = 10^0 (1) down to 10^-50, every exponent

device = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(RANDOM_SEED)
transform = Compose([ToTensor()])
train_dataset = LocalZipCIFAR10Dataset(ZIP_PATH, train=True)


def discover_checkpoints():
    """poi_method -> checkpoint path, matched against the known method list.
    Filenames are asset_<model>_<poi_method>_<threshold>.pth -- the threshold
    suffix is ignored. Matching requires the character right after
    "<poi_method>_" to start a number, so "backdoor" can't accidentally match
    "backdoor_all2all_..." (a different method that happens to share "backdoor"
    as a prefix). If more than one file matches, the most recently saved wins."""
    found = {}
    for poi_method in ALL_POI_METHODS:
        matches = glob.glob(f"{ASSET_BEST_DIR}/asset_{MODEL_NAME}_{poi_method}_[0-9]*.pth")
        if matches:
            found[poi_method] = max(matches, key=os.path.getmtime)
    return found


def compute_metrics(poi_res, clean_res, n_poi, eps):
    total = poi_res + clean_res
    t = get_t(total, eps)
    tp = n_poi - np.where(np.array(poi_res) < t)[0].shape[0]
    fp = len(clean_res) - np.where(np.array(clean_res) < t)[0].shape[0]
    fn = np.where(np.array(poi_res) < t)[0].shape[0]
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr_rate = fp / len(clean_res) if len(clean_res) > 0 else 0.0
    return {"TP": tp, "FP": fp, "FN": fn, "Recall/TPR": recall,
            "Precision": precision, "F1": f1, "FPR": fpr_rate}


checkpoints = discover_checkpoints()
print(f"Found {len(checkpoints)} checkpoint(s): {sorted(checkpoints)}")

all_rows = []
best_rows = []

for poi_method, ckpt_path in sorted(checkpoints.items()):
    print(f"\n=== {poi_method} ({os.path.basename(ckpt_path)}) ===")
    tar_lab_arg = (FLIP_SRC_LAB, TAR_LAB) if poi_method == "flipping_label" else TAR_LAB

    train_poi_set, o_poi_idx = poi_dataset(
        train_dataset, poi_method=poi_method, transform=transform,
        poi_rates=0.05, random_seed=RANDOM_SEED, tar_lab=tar_lab_arg,
    )

    model = ResNet18().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # expensive part -- done ONCE per method, same as loading the model once in the notebook
    poi_res, clean_res = get_result(model, train_poi_set, o_poi_idx)
    true_label = [1] * len(poi_res) + [0] * len(clean_res)
    pred_label = poi_res + clean_res
    auc = roc_auc_score(true_label, pred_label)
    print(f"  AUC={auc:.4f}")

    best_f1, best_exp, best_metrics = -1.0, None, None
    for exp in EXPONENTS:
        eps = 10.0 ** -exp
        m = compute_metrics(poi_res, clean_res, len(o_poi_idx), eps)
        all_rows.append({"POI_METHOD": poi_method, "Epsilon": eps, "AUC": auc, **m})
        print(f"    eps=1e-{exp:02d}  tp={m['TP']:5d} fp={m['FP']:6d} fn={m['FN']:5d}  F1={m['F1']:.4f}")
        if m["F1"] > best_f1:
            best_f1, best_exp, best_metrics = m["F1"], exp, m

    print(f"  ** best eps=1e-{best_exp:02d}  F1={best_f1:.4f}  "
          f"(tp={best_metrics['TP']}, fp={best_metrics['FP']}, fn={best_metrics['FN']})")
    best_rows.append({"POI_METHOD": poi_method, "Epsilon": 10.0 ** -best_exp, "AUC": auc, **best_metrics})

pd.DataFrame(all_rows).to_csv(RESULTS_CSV, index=False)
print(f"\nFull sweep ({len(all_rows)} rows) saved to {RESULTS_CSV}")

print("\n=== BEST EPS PER METHOD ===")
print(pd.DataFrame(best_rows).to_string(index=False))
