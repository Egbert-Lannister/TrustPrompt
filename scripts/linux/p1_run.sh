#!/usr/bin/env bash
# Defaults to the one-hour P1 pilot only. Full runs require a reviewed pilot
# report and explicit user approval; then set TP_P1_ALLOW_FULL_TRAIN=1.
set -euo pipefail
cd "$(dirname "$0")/../.."
. "${TP_P1_VENV:-.venv-p1}/bin/activate"
latest() {
  python -c "import pathlib,sys; d=sorted(p for p in pathlib.Path('results').glob('p1_'+sys.argv[1]+'_*') if p.is_dir()); print(d[-1] if d else '')" "$1"
}
case "${1:-pilot}" in
  pilot)
    command -v timeout >/dev/null || { echo "GNU timeout is required for the pilot wall limit." >&2; exit 2; }
    # Includes process/model startup; trainer reserves 60 seconds for saving.
    # This bounds the process, not instance billing or GPU-kernel time.
    before=$(latest f1)
    status=0
    timeout --signal=INT --kill-after=30s 60m python -m scripts.train_omnidc --variant f1 --pilot-minutes 60 || status=$?
    d=$(latest f1)
    if [ -n "$d" ] && [ "$d" != "$before" ]; then
      python -c "import json,sys; r=json.load(open(sys.argv[1]+'/receipt.json')); print(json.dumps(dict(stage=r.get('stage'),pilot=r.get('pilot'),error=r.get('error')),indent=2))" "$d"
    else
      echo "No new run directory was created; inspect the error above." >&2
    fi
    echo "Pilot stopped. Review the report and obtain user approval before any full training."
    exit "$status" ;;
  train)
    if [ "${TP_P1_ALLOW_FULL_TRAIN:-0}" != "1" ]; then
      echo "Full training is disabled. After explicit user approval set TP_P1_ALLOW_FULL_TRAIN=1." >&2
      exit 2
    fi
    for v in f0 f1; do
      d=$(latest "$v")
      if [ -n "$d" ]; then
        stage=$(python -c "import json,sys; print(json.load(open(sys.argv[1]+'/receipt.json')).get('stage',''))" "$d")
        if [ "$stage" = training_complete ]; then echo "$v done: $d"; continue; fi
        # Missing/invalid checkpoints are errors, never silent fresh runs.
        python -m scripts.train_omnidc --variant "$v" --resume "$d"
      else
        python -m scripts.train_omnidc --variant "$v"
      fi
    done
    ;;
  *) echo "usage: $0 [pilot|train]"; exit 1 ;;
esac
