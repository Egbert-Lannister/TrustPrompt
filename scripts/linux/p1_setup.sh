#!/usr/bin/env bash
# One-time setup of a rented Linux GPU machine for the mainline P1 (docs/plan_omnidc.md sec. 7, 17).
# Run from the TrustPrompt checkout (copied from your machine: git bundle / zip of the tracked files).
# No private data is uploaded: the OMNI-DC weights come from Hugging Face and the Hypersim frames are fetched again
# from Apple's public release with the same pinned selection, then checked against the hash of your local list.
#   bash scripts/linux/p1_setup.sh [CUDA wheel index, default cu126]
set -euo pipefail
cd "$(dirname "$0")/../.."
INDEX="${1:-cu126}"
FRAMES_SHA256=998367e5713ccd5a1adcb55aef425bcb200dc58527cb947e3f304f9d248c6edc    # data/source_cache/hypersim_v1/frames_v1.json on the laptop
python3 -m venv .venv-p1
. .venv-p1/bin/activate
pip install --upgrade pip
pip install torch==2.11.0 torchvision==0.26.0 --index-url "https://download.pytorch.org/whl/${INDEX}"
pip install -r requirements-local.txt huggingface_hub==0.36.2 safetensors==0.6.2 \
    $(python3 -c "import json; print(' '.join(json.load(open('configs/baselines/registry.json'))['pip']))")
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print(torch.__version__, torch.cuda.get_device_name())"
python -m scripts.setup_baselines --models omnidc
python -m scripts.hypersim_fetch plan --max-gb 40
python -m scripts.hypersim_fetch semantic --workers 16
python -m scripts.hypersim_fetch select
echo "${FRAMES_SHA256}  data/source_cache/hypersim_v1/frames_v1.json" | sha256sum -c -
python -m scripts.hypersim_fetch frames --workers 16
python -m pytest -q tests/test_omnidc_train.py tests/test_omnidc_gate.py tests/test_hypersim.py tests/test_hypersim_condition.py tests/test_onesided.py
echo "P1 SETUP DONE. Next: bash scripts/linux/p1_run.sh pilot"
