#!/usr/bin/env bash
#
# SLURM — Payload trainer
#
#SBATCH --job-name=training
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=36G
#SBATCH --time=12:00:00
#SBATCH --output=Results/slurm-%j.out
#SBATCH --error=Results/slurm-%j.err

set -euo pipefail

# --- Paths (persistent venv lives OUTSIDE ~/Payload) ---
VENV_BASE="/vol/fob-vol1/mi23/ziglowsa/venvs"
VENV_NAME="payload_venv"
VENV="${VENV_BASE}/${VENV_NAME}"
CACHE_BASE="${VENV_BASE}/.cache"

mkdir -p "${VENV_BASE}" "${CACHE_BASE}" "${HOME}/Payload/logs"

module purge >/dev/null 2>&1 || true
module load python/3.10 >/dev/null 2>&1 || true


if [ ! -x "${VENV}/bin/python" ]; then
  python3 -m venv "${VENV}"
fi
source "${VENV}/bin/activate"
-
export TMPDIR="${CACHE_BASE}/tmp"
export PIP_CACHE_DIR="${CACHE_BASE}/pip"
export TORCH_HOME="${CACHE_BASE}/torch"
export HF_HOME="${CACHE_BASE}/hf"
export CUDA_CACHE_PATH="${CACHE_BASE}/.nv/ComputeCache"
mkdir -p "${TMPDIR}" "${PIP_CACHE_DIR}" "${TORCH_HOME}" "${HF_HOME}" "${CUDA_CACHE_PATH}"

python -m pip install -U pip wheel setuptools packaging --no-cache-dir


MISSING_IMPORTS="$(python - <<'PY'
import importlib

required = [
    "sys",
    "time",
    "datetime",
    "yaml",
    "pandas",
    "sklearn.model_selection",
    "seaborn",
]

missing = []
for mod in required:
    try:
        importlib.import_module(mod)
    except Exception:
        missing.append(mod)

print(" ".join(missing))
PY
)"

if [ -n "${MISSING_IMPORTS}" ]; then
  echo "Missing imports (venv): ${MISSING_IMPORTS}"

  PIP_PKGS=()
  for m in ${MISSING_IMPORTS}; do
    case "${m}" in
      yaml) PIP_PKGS+=("PyYAML") ;;
      pandas) PIP_PKGS+=("pandas") ;;
      sklearn.model_selection) PIP_PKGS+=("scikit-learn") ;;
      seaborn) PIP_PKGS+=("seaborn") ;;
      # stdlib should never be installed
      sys|time|datetime) ;;
      *) echo "Warning: no pip mapping for '${m}'" >&2 ;;
    esac
  done

  if [ ${#PIP_PKGS[@]} -gt 0 ]; then
    UNIQUE_PKGS=()
    for p in "${PIP_PKGS[@]}"; do
      skip=false
      for u in "${UNIQUE_PKGS[@]}"; do
        if [ "${u}" = "${p}" ]; then skip=true; break; fi
      done
      ${skip} || UNIQUE_PKGS+=("${p}")
    done

    echo "Installing into venv via pip: ${UNIQUE_PKGS[*]}"
    python -m pip install --no-cache-dir "${UNIQUE_PKGS[@]}"
  fi
else
  echo "All required imports already available in venv."
fi

TORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"
TORCH_TARGET="2.4.1"
TV_TARGET="0.19.1"
TA_TARGET="2.4.1"

VERSIONS="$(python - <<'PY'
import importlib

def ver(mod: str) -> str:
    try:
        m = importlib.import_module(mod)
        v = getattr(m, "__version__", "")
        return (v.split('+')[0] if v else "")
    except Exception:
        return ""

print(ver("torch"), ver("torchvision"), ver("torchaudio"))
PY
)"

read -r INSTALLED_TORCH INSTALLED_TV INSTALLED_TA <<< "${VERSIONS}"

NEED_TORCH=false
[ "${INSTALLED_TORCH}" != "${TORCH_TARGET}" ] && NEED_TORCH=true
[ "${INSTALLED_TV}" != "${TV_TARGET}" ] && NEED_TORCH=true
[ "${INSTALLED_TA}" != "${TA_TARGET}" ] && NEED_TORCH=true

if ${NEED_TORCH}; then
  echo "Ensuring torch stack in venv (torch=${TORCH_TARGET}, torchvision=${TV_TARGET}, torchaudio=${TA_TARGET})"
  python -m pip install --no-cache-dir \
    --index-url "${TORCH_INDEX_URL}" \
    "torch==${TORCH_TARGET}" \
    "torchvision==${TV_TARGET}" \
    "torchaudio==${TA_TARGET}"
fi

python - <<'PY'
import importlib, sys

required = [
    "sys",
    "time",
    "datetime",
    "yaml",
    "pandas",
    "sklearn.model_selection",
    "torch",
    "torchvision",
    "seaborn",
]

missing = []
for mod in required:
    try:
        importlib.import_module(mod)
    except Exception as e:
        missing.append((mod, str(e)))

if missing:
    for mod, err in missing:
        print(f"FATAL: cannot import {mod}: {err}")
    sys.exit(1)

print("Import check passed (including torchvision).")
PY

python - <<'PY'
import os, subprocess
print("=== ENV ===")
print("HOSTNAME:", os.uname().nodename)
print("PARTITION:", os.environ.get("SLURM_JOB_PARTITION"))
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("PWD:", os.getcwd())
try:
    import numpy as np
    print("numpy:", np.__version__)
except Exception as e:
    print("numpy import failed:", e)
try:
    import torch
    print("torch:", torch.__version__, "cuda:", torch.version.cuda, "is_available:", torch.cuda.is_available())
except Exception as e:
    print("torch import failed:", e)
try:
    import torchvision
    print("torchvision:", torchvision.__version__)
except Exception as e:
    print("torchvision import failed:", e)
try:
    print("nvidia-smi -L:", subprocess.check_output(["nvidia-smi","-L"], text=True).strip())
except Exception as e:
    print("nvidia-smi error:", e)
PY

cd "${HOME}/Payload"
python scripts/train.py
