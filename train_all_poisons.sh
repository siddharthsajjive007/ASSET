#!/bin/bash
# Trains train_backdoored_model.py for every poisoning method that doesn't yet have
# a checkpoint saved. Safe to re-run any time -- methods that already have a
# checkpoint are skipped, so this is the one command to run after adding a new
# poisoning method to new_poi_util.py: it'll only train what's actually missing.
set -uo pipefail
cd /home/siddarth/ASSET
source /home/siddarth/UV-ENV/bin/activate

CHECKPOINT_DIR="/home/HDD/ATAF/siddharth/ASSET/trained_models/poisoned_models"

# Keep this in sync with the poi_method branches in new_poi_util.py's poi_dataset().
methods=(backdoor backdoor_all2all noisy_label flipping_label clean_label_narcissus
         noisy_all2one noisy_all2all wanet blended badnets low_frequency ssba
         inputaware ctrl)

summary_file="logs/train_all_poisons_summary.txt"
: > "$summary_file"

for m in "${methods[@]}"; do
    ckpt="${CHECKPOINT_DIR}/resnet18_cifar10_${m}.pth"
    if [ -f "$ckpt" ]; then
        echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] Skipping POI_METHOD=$m -- checkpoint already exists at $ckpt ==="
        echo "$m: SKIPPED (already trained)" >> "$summary_file"
        continue
    fi

    echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] Starting POI_METHOD=$m ==="
    # train_backdoored_model.py does its own logging to logs/train_<method>.log internally.
    POI_METHOD="$m" python train_backdoored_model.py
    status=$?
    if [ "$status" -eq 0 ]; then
        echo "$m: SUCCESS" >> "$summary_file"
        echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] Finished POI_METHOD=$m (success) ==="
    else
        echo "$m: FAILED (exit $status)" >> "$summary_file"
        echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] POI_METHOD=$m FAILED (exit $status) -- continuing to next method ==="
    fi
done

echo "=== ALL METHODS COMPLETE ==="
echo "Summary:"
cat "$summary_file"
