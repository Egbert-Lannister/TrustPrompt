#!/usr/bin/env bash
# P1 setup only: no training, model construction, private-data upload or account creation.
# CPU preparation keeps the target CUDA wheels, so attaching a GPU reuses the SAME environment.
# Examples (Ubuntu 22.04 / Python 3.12; code/data may live on a separate 50 GB data disk):
#   TP_P1_VENV=/root/trustprompt-envs/p1-cu126 bash scripts/linux/p1_setup.sh --cpu-only
#   TP_P1_VENV=/root/trustprompt-envs/p1-cu126 bash scripts/linux/p1_setup.sh --cpu-only --stage env
#   TP_P1_VENV=/root/trustprompt-envs/p1-cu126 bash scripts/linux/p1_setup.sh --cpu-only --from semantic
# Every chosen stage revalidates inputs; success markers never bypass cache/hash verification.
set -Eeuo pipefail

usage() {
  cat <<'USAGE'
Usage: bash scripts/linux/p1_setup.sh [cu126] [options]
  --cpu-only           Prepare without GPU; no CUDA check, data workers <= 2.
                       Installs the requested CUDA wheel build for later GPU use.
  --index NAME         PyTorch wheel index (default cu126; positional form retained).
  --workers N          Data-download workers (default 2 CPU / 16 GPU).
  --stage NAME         Run one stage; repeat or use --stages env,model,plan,... .
  --from NAME          Resume at this stage and continue in dependency order.
  --help               Show help without installing or creating directories.
Stages: env model plan semantic select frames tests. Default: all stages.
Individual stages require their preceding outputs. No second environment is created.
TP_P1_VENV selects the single venv (default .venv-p1); TP_P1_PYTHON selects its base Python.
TP_P1_TORCH_INDEX_URL optionally overrides the torch wheel index URL (recorded in setup.json);
the default remains https://download.pytorch.org/whl/<index> with the same exact package pins.
Use a system-disk venv when the checkout/data live on a separate 50 GB data disk.
Logs/receipts: results/p1_setup/<UTC timestamp>-<pid>/. Re-run failed stages to resume caches.
The plan stage retains the conservative full-download free-space check; after a partial
download, resume --from semantic or --from frames instead of rerunning the plan stage.
USAGE
}

INDEX=cu126
CPU_ONLY=0
WORKERS="${TP_P1_DATA_WORKERS:-}"
BASE_PYTHON="${TP_P1_PYTHON:-python3}"
REQUESTED=()
FROM_STAGE=""
POSITIONAL_INDEX=0
ALL_STAGES=(env model plan semantic select frames tests)
while (($#)); do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --cpu-only) CPU_ONLY=1; shift ;;
    --index|--workers|--stage|--stages|--from)
      if (($# < 2)) || [[ -z "$2" ]]; then echo "Missing value for $1" >&2; exit 2; fi
      case "$1" in
        --index) INDEX="$2" ;;
        --workers) WORKERS="$2" ;;
        --stage|--stages)
          IFS=',' read -r -a PARTS <<< "$2"
          REQUESTED+=("${PARTS[@]}") ;;
        --from) FROM_STAGE="$2" ;;
      esac
      shift 2 ;;
    --*) echo "Unknown option: $1" >&2; exit 2 ;;
    *)
      if ((POSITIONAL_INDEX)); then echo "Unexpected argument: $1" >&2; exit 2; fi
      INDEX="$1"; POSITIONAL_INDEX=1; shift ;;
  esac
done
[[ "$INDEX" =~ ^(cu[0-9]+|cpu)$ ]] || { echo "Invalid PyTorch wheel index: $INDEX" >&2; exit 2; }
TORCH_INDEX_URL="${TP_P1_TORCH_INDEX_URL:-https://download.pytorch.org/whl/${INDEX}}"
if [[ -z "$WORKERS" ]]; then
  if ((CPU_ONLY)); then WORKERS=2; else WORKERS=16; fi
fi
[[ "$WORKERS" =~ ^[1-9][0-9]*$ ]] || { echo "workers must be a positive integer" >&2; exit 2; }
if ((CPU_ONLY && WORKERS > 2)); then echo "--cpu-only allows at most 2 download workers" >&2; exit 2; fi
if [[ -n "$FROM_STAGE" && ${#REQUESTED[@]} -gt 0 ]]; then
  echo "Use --from or --stage/--stages, not both" >&2; exit 2
fi
for stage in "${REQUESTED[@]}" ${FROM_STAGE:+"$FROM_STAGE"}; do
  case "$stage" in env|model|plan|semantic|select|frames|tests) ;; *) echo "Unknown stage: $stage" >&2; exit 2 ;; esac
done
STAGES=()
if [[ -n "$FROM_STAGE" ]]; then
  found=0
  for stage in "${ALL_STAGES[@]}"; do
    [[ "$stage" != "$FROM_STAGE" ]] || found=1
    if ((found)); then STAGES+=("$stage"); fi
  done
elif ((${#REQUESTED[@]})); then
  for stage in "${ALL_STAGES[@]}"; do
    for wanted in "${REQUESTED[@]}"; do
      if [[ "$stage" == "$wanted" ]]; then STAGES+=("$stage"); break; fi
    done
  done
else
  STAGES=("${ALL_STAGES[@]}")
fi

cd "$(dirname "$0")/../.."
ROOT="$PWD"
VENV="${TP_P1_VENV:-$ROOT/.venv-p1}"
[[ "$VENV" = /* ]] || VENV="$ROOT/$VENV"
PYTHON="$VENV/bin/python"
# Old Windows raw SHA256 retained as provenance only (CRLF is different on Linux):
# 998367e5713ccd5a1adcb55aef425bcb200dc58527cb947e3f304f9d248c6edc
FRAMES_CANONICAL_SHA256=ba4768d0f8a0af30f30bc973148d1a6befc830ebf6cabf2becbd28193985cd2f
export PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_INPUT=1 PIP_PROGRESS_BAR=off
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
if ((CPU_ONLY)); then
  export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
  # Avoid Xet's parallel buffer pressure on the 2 GB staging instance.
  export HF_HUB_DISABLE_XET=1
fi
export TP_SETUP_CPU_ONLY="$CPU_ONLY" TP_SETUP_WORKERS="$WORKERS" TP_SETUP_INDEX="$INDEX"
export TP_SETUP_TORCH_INDEX_URL="$TORCH_INDEX_URL"
export TP_P1_VENV="$VENV" TP_SETUP_FRAMES_SHA256="$FRAMES_CANONICAL_SHA256"

command -v "$BASE_PYTHON" >/dev/null || { echo "Base Python not found: $BASE_PYTHON" >&2; exit 1; }
command -v flock >/dev/null || { echo "flock (util-linux) is required for safe staged setup" >&2; exit 1; }
mkdir -p results/p1_setup
exec 9>results/p1_setup/setup.lock
flock -n 9 || { echo "Another P1 setup is running; do not run concurrent setup/download stages" >&2; exit 1; }
RUN_DIR="$ROOT/results/p1_setup/$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$RUN_DIR"
export TP_SETUP_RUN_DIR="$RUN_DIR"
exec > >(tee -a "$RUN_DIR/run.log") 2>&1
CURRENT_STAGE=preflight
on_error() {
  local status=$?
  printf '%s\t%s\tfailed\t%s\n' "$(date -u +%FT%TZ)" "$CURRENT_STAGE" "$status" >> "$RUN_DIR/progress.tsv"
  echo "P1 SETUP FAILED: stage=$CURRENT_STAGE exit=$status; logs=$RUN_DIR; caches preserved."
  if [[ "$CURRENT_STAGE" != preflight ]]; then
    echo "Resume with --stage $CURRENT_STAGE, or --from $CURRENT_STAGE for remaining stages."
  fi
  exit "$status"
}
trap on_error ERR
printf 'P1 SETUP: cpu_only=%s index=%s workers=%s venv=%s stages=%s\n' "$CPU_ONLY" "$INDEX" "$WORKERS" "$VENV" "${STAGES[*]}"

# Validate public pinned inputs before downloads; record source versions and actual cgroup limits.
"$BASE_PYTHON" - "${STAGES[@]}" <<'PY'
import hashlib, json, os, pathlib, re, shutil, subprocess, sys
root = pathlib.Path.cwd()
reg = json.loads((root / "configs/baselines/registry.json").read_text(encoding="utf-8"))
spec = reg["models"]["omnidc"]
assert spec["repo"] == "https://github.com/princeton-vl/OMNI-DC"
assert re.fullmatch(r"[0-9a-f]{40}", spec["commit"])
for weight in spec["hf"]:
    assert weight["repo"] in {"zuoym15/OMNI-DC", "Rain729/Prior-Depth-Anything"}
    assert re.fullmatch(r"[0-9a-f]{40}", weight["revision"])
    for name, (size, digest) in weight["files"].items():
        assert size > 0 and re.fullmatch(r"[0-9a-f]{64}", digest or ""), name
selection = json.loads((root / "configs/hypersim/selection_v1.json").read_text(encoding="utf-8"))
assert selection["source"]["repo"] == "https://github.com/apple/ml-hypersim"
assert re.fullmatch(r"[0-9a-f]{40}", selection["source"]["commit"])
assert selection["source"]["files"] and all(re.fullmatch(r"[0-9a-f]{64}", h) for h in selection["source"]["files"].values())
assert selection["url_template"] == "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/{scene}.zip"
assert all(re.fullmatch(r"ai_[0-9]{3}_[0-9]{3}", scene) for scene in selection["scenes"])
paths = ["scripts/linux/p1_setup.sh", "scripts/setup_baselines.py", "scripts/hypersim_fetch.py",
         "baselines/completion.py", "tp/data/remote_zip.py", "requirements-local.txt",
         "configs/baselines/registry.json", "configs/hypersim/selection_v1.json"]
hashes = {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in paths}
def read_cgroup(name):
    p = pathlib.Path("/sys/fs/cgroup") / name
    return p.read_text().strip() if p.is_file() else None
def git(*args):
    p = subprocess.run(["git", *args], capture_output=True, text=True)
    return p.stdout.strip() if p.returncode == 0 else None
record = dict(schema="trustprompt.p1_setup.v2", stages=sys.argv[1:], cpu_only=os.environ["TP_SETUP_CPU_ONLY"] == "1",
              workers=int(os.environ["TP_SETUP_WORKERS"]), wheel_index=os.environ["TP_SETUP_INDEX"],
              torch_index_url=os.environ["TP_SETUP_TORCH_INDEX_URL"],
              venv=os.environ["TP_P1_VENV"], source_sha256=hashes, code_commit=git("rev-parse", "HEAD"),
              code_status=git("status", "--porcelain=v1", "--untracked-files=no"),
              model_source=spec, data_source=selection["source"],
              frames_canonical_sha256=os.environ["TP_SETUP_FRAMES_SHA256"],
              cgroup_cpu_max=read_cgroup("cpu.max"), cgroup_memory_max=read_cgroup("memory.max"),
              checkout_free_gb=shutil.disk_usage(root).free / 1e9)
out = pathlib.Path(os.environ["TP_SETUP_RUN_DIR"]) / "setup.json"
out.write_text(json.dumps(record, indent=2), encoding="utf-8")
print("Pinned public inputs recorded:", out)
print("cgroup CPU/RAM:", record["cgroup_cpu_max"], record["cgroup_memory_max"])
PY

import_preflight() {
  "$PYTHON" - <<'PY'
import importlib.metadata as md, json, os, pathlib
import torch, torchvision, torchvision.ops, cv2
assert torch.__version__.split("+")[0] == "2.11.0", torch.__version__
assert torchvision.__version__.split("+")[0] == "0.26.0", torchvision.__version__
assert md.version("opencv-python-headless") == "4.11.0.86"
# Exercise compiled CPU operators without instantiating OMNI-DC or loading checkpoints.
boxes = torch.tensor([[0., 0., 2., 2.], [0., 0., 2., 2.]])
assert torchvision.ops.nms(boxes, torch.tensor([.9, .8]), .5).tolist() == [0]
cpu_only = os.environ["TP_SETUP_CPU_ONLY"] == "1"
device = "not checked (CPU staging)"
if not cpu_only:
    assert torch.cuda.is_available(), "No CUDA device; use --cpu-only while staging, then re-run after GPU attachment"
    device = torch.cuda.get_device_name(0)
record = dict(torch=torch.__version__, torchvision=torchvision.__version__, cv2=cv2.__version__,
              opencv_python_headless=md.version("opencv-python-headless"),
              cuda_build=torch.version.cuda, device=device, cpu_operator_check="passed", model_loaded=False)
(pathlib.Path(os.environ["TP_SETUP_RUN_DIR"]) / "imports.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
print("IMPORT PREFLIGHT", json.dumps(record))
PY
}

stage_env() {
  if [[ ! -x "$PYTHON" ]]; then
    if [[ -e "$VENV" ]]; then echo "$VENV exists but is not a usable venv; refusing to overwrite it" >&2; return 1; fi
    "$BASE_PYTHON" -c 'import sys; assert (3, 10) <= sys.version_info[:2] <= (3, 12), "Pinned packages need Python 3.10-3.12; use TP_P1_PYTHON=python3.12"'
    "$BASE_PYTHON" -m venv "$VENV"
  fi
  "$PYTHON" -m pip --version
  "$PYTHON" -m pip install --no-cache-dir torch==2.11.0 torchvision==0.26.0 --index-url "$TORCH_INDEX_URL"
  mapfile -t PINS < <("$PYTHON" -c 'import json; print("\n".join(json.load(open("configs/baselines/registry.json"))["pip"]))')
  "$PYTHON" -m pip install --no-cache-dir -r requirements-local.txt huggingface_hub==0.36.2 safetensors==0.6.2 \
    opencv-python-headless==4.11.0.86 "${PINS[@]}"
  "$PYTHON" -m pip check
  "$PYTHON" -m pip freeze > "$RUN_DIR/packages.txt"
  import_preflight
}

stage_model() { "$PYTHON" -m scripts.setup_baselines --models omnidc; }
stage_plan() { "$PYTHON" -m scripts.hypersim_fetch plan --max-gb 40; }
stage_semantic() { "$PYTHON" -m scripts.hypersim_fetch semantic --workers "$WORKERS"; }
verify_frames() {
  "$PYTHON" - <<'PY'
import hashlib, json, os, pathlib
p = pathlib.Path("data/source_cache/hypersim_v1/frames_v1.json")
obj = json.loads(p.read_text(encoding="utf-8"))
canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
actual = hashlib.sha256(canonical).hexdigest()
expected = os.environ["TP_SETUP_FRAMES_SHA256"]
assert actual == expected, f"Selected frames differ from pinned public selection: {actual} != {expected}"
print("SELECTED FRAMES canonical SHA256 OK:", actual)
PY
}
stage_select() { "$PYTHON" -m scripts.hypersim_fetch select; verify_frames; }
stage_frames() { verify_frames; "$PYTHON" -m scripts.hypersim_fetch frames --workers "$WORKERS"; }
stage_tests() {
  # P1 does not use P2's optional hinge module. Its unchanged exhaustive 640-variable
  # finite-difference test takes >=1280 DDI forward solves; keep it available for P2,
  # but do not run it on the 0.5 CPU staging instance. Other hinge checks still run.
  "$PYTHON" -m pytest -q tests/test_omnidc_train.py tests/test_omnidc_gate.py tests/test_hypersim.py \
    tests/test_hypersim_condition.py tests/test_onesided.py tests/test_remote_zip.py tests/test_provenance.py \
    --deselect=tests/test_onesided.py::test_gradients_through_omnidc_ddi_with_the_hinge
}

if [[ "${STAGES[0]}" != env ]]; then
  [[ -x "$PYTHON" ]] || { echo "Missing $PYTHON; run --stage env first with the same TP_P1_VENV" >&2; exit 1; }
  import_preflight
fi
for stage in "${STAGES[@]}"; do
  CURRENT_STAGE="$stage"
  printf '%s\t%s\tstarted\n' "$(date -u +%FT%TZ)" "$stage" >> "$RUN_DIR/progress.tsv"
  echo "=== P1 setup stage: $stage ($(date -u +%FT%TZ))"
  "stage_$stage"
  printf '%s\t%s\tcomplete\n' "$(date -u +%FT%TZ)" "$stage" >> "$RUN_DIR/progress.tsv"
  printf '%s\n' "$(date -u +%FT%TZ)" > "$RUN_DIR/$stage.done"
done
echo "P1 SETUP STAGES COMPLETE: ${STAGES[*]}; receipt/logs: $RUN_DIR"
if ((CPU_ONLY)); then
  echo "CPU preparation only: no training or full-model load was attempted."
  echo "After GPU attachment, use the same TP_P1_VENV and run --stage tests for the CUDA/import check."
else
  echo "Before the separately authorized pilot, ensure all seven setup stages have completed."
  echo "Next: TP_P1_VENV='$VENV' bash scripts/linux/p1_run.sh pilot"
fi
