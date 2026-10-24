#!/usr/bin/env bash
# Mainline P1 on the rented GPU (docs/plan_omnidc.md sec. 7, 17). Run after p1_setup.sh.
#   bash scripts/linux/p1_run.sh pilot    # 60 min of f1: measured throughput + GPU-hour estimate; stop and report
#   bash scripts/linux/p1_run.sh train    # f0 then f1, full runs (resumable: re-run the same command)
# Copy results/p1_*/checkpoints/final.pt back to your machine for the evaluation (it runs there, on the dev sets).
set -euo pipefail
cd "$(dirname "$0")/../.."
. .venv-p1/bin/activate
latest() { ls -d results/p1_"$1"_* 2>/dev/null | sort | tail -1; }
case "${1:-}" in
  pilot)
    python -m scripts.train_omnidc --variant f1 --pilot-minutes 60
    python -c "import json,glob; d=sorted(glob.glob('results/p1_f1_*'))[-1]; print(json.load(open(d+'/receipt.json')).get('pilot'))" ;;
  train)
    for v in f0 f1; do
      d=$(latest "$v")
      if [ -n "$d" ] && grep -q '"training_complete"' "$d/receipt.json"; then echo "$v done: $d"; continue; fi
      if [ -n "$d" ] && grep -q '"stage": "training"\|"failed"' "$d/receipt.json" && [ -f "$d/checkpoints/last.pt" ]; then
        python -m scripts.train_omnidc --variant "$v" --resume "$d"
      else
        python -m scripts.train_omnidc --variant "$v"
      fi
    done
    echo "P1 TRAINING DONE: copy results/p1_f0_*/checkpoints/final.pt and results/p1_f1_*/checkpoints/final.pt back." ;;
  *) echo "usage: $0 pilot|train"; exit 1 ;;
esac
