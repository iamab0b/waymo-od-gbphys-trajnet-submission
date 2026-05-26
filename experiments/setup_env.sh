#!/bin/bash
# ---------------------------------------------------------------------------
# GB-Phys TrajNet  --  Anaconda/Miniforge conda environment setup
#
# Handles all known dependency conflicts:
#   - numpy must be installed BEFORE tensorflow (ABI ordering)
#   - absl-py, google-auth, Pillow, scikit-learn pinned BEFORE waymo SDK
#     to prevent the pre-installed newer versions from causing conflicts
#   - scipy installed early (required by jax 0.4.13)
#   - typing_extensions installed from a local wheel to bypass TF metadata
#   - jaxlib 0.4.13 from the JAX GCS bucket (not on PyPI)
#   - tensorflow_probability 0.21.0 is a waymo SDK runtime requirement
#   - CUDA version auto-detected from loaded modules then nvidia-smi
#
# Usage:
#   bash experiments/setup_env.sh                        # auto-detect CUDA
#   bash experiments/setup_env.sh --force                # wipe & recreate (SLURM)
#   bash experiments/setup_env.sh --force --cuda cu121   # force CUDA build
#
# CUDA build tags: cu128  cu126  cu124  cu121  cu118  cpu
# This cluster (Zaratan) has CUDA toolkit 12.3.0 -> use cu121
# ---------------------------------------------------------------------------

set -eo pipefail

ENV_NAME="waymo-trajnet"
PYTHON_VERSION="3.10"
FORCE=""
CUDA_OVERRIDE=""

# Parse arguments
for arg in "$@"; do
    case "${arg}" in
        --force) FORCE="--force" ;;
        cu128|cu126|cu124|cu121|cu118|cpu) CUDA_OVERRIDE="${arg}" ;;
    esac
done

# Support:  --cuda cu121
NEXT_IS_CUDA=0
for arg in "$@"; do
    if [[ ${NEXT_IS_CUDA} -eq 1 ]]; then
        CUDA_OVERRIDE="${arg}"
        NEXT_IS_CUDA=0
    fi
    if [[ "${arg}" == "--cuda" ]]; then
        NEXT_IS_CUDA=1
    fi
done

# ── Path to your conda installation ──────────────────────────────────────────
CONDA_BASE="/home/smehta22/scratch.cmsc472/conda/miniforge3-26.1.1"

echo "======================================================"
echo " GB-Phys TrajNet  --  conda environment setup"
echo " Environment : ${ENV_NAME}"
echo " Python      : ${PYTHON_VERSION}"
echo " Conda base  : ${CONDA_BASE}"
echo "======================================================"

# ── Load CUDA module (provides libcuda, nvcc, nvidia-smi on compute nodes) ───
if command -v module &>/dev/null; then
    echo "Loading CUDA 12.3.0 module..."
    module load cuda/12.3.0 2>/dev/null || \
    module load cuda-new/zen2/12.3.0 2>/dev/null || \
    echo "  [WARN] Could not load CUDA module — will use system CUDA if present."
fi

# ── Initialize conda for this shell session ───────────────────────────────────
# set +eu because conda.sh uses unbound variables that break strict mode
if [[ ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    echo "ERROR: ${CONDA_BASE}/etc/profile.d/conda.sh not found" >&2
    exit 1
fi
set +eu
source "${CONDA_BASE}/etc/profile.d/conda.sh"
set -eo pipefail
echo "Conda: $(conda --version)"

# ── Remove / reuse existing environment ──────────────────────────────────────
if conda env list | grep -q "^${ENV_NAME}[[:space:]]"; then
    if [[ "${FORCE}" == "--force" ]]; then
        echo "Removing existing '${ENV_NAME}' (--force)..."
        conda env remove -n "${ENV_NAME}" -y
    else
        echo "Environment '${ENV_NAME}' already exists."
        read -rp "Remove and recreate? [y/N] " yn
        case "$yn" in
            [yY]) conda env remove -n "${ENV_NAME}" -y ;;
            *)
                echo "Keeping existing environment."
                set +eu
                conda activate "${ENV_NAME}"
                set -eo pipefail
                exit 0
                ;;
        esac
    fi
fi

# ── Create fresh environment ──────────────────────────────────────────────────
echo ""
echo "Creating conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
set +eu
conda activate "${ENV_NAME}"
set -eo pipefail
echo "Active Python: $(which python)  $(python --version)"

PIP="python -m pip"
${PIP} install --upgrade pip setuptools wheel

# ── Detect CUDA version ───────────────────────────────────────────────────────
detect_torch_index() {
    if [[ -n "${CUDA_OVERRIDE:-}" ]]; then
        echo "${CUDA_OVERRIDE}"; return
    fi

    CUDA_VER=""

    # Check loaded CUDA module
    if command -v module &>/dev/null; then
        CUDA_VER=$(module list 2>&1 \
            | grep -oE 'cuda[^/]*/[0-9]+\.[0-9]+' \
            | grep -oE '[0-9]+\.[0-9]+' \
            | head -1 || echo "")
    fi

    # Fallback: nvidia-smi
    if [[ -z "${CUDA_VER:-}" ]] && command -v nvidia-smi &>/dev/null; then
        CUDA_VER=$(nvidia-smi 2>/dev/null \
            | grep "CUDA Version" \
            | sed 's/.*CUDA Version: //' \
            | awk '{print $1}' || echo "")
    fi

    # Fallback: nvcc
    if [[ -z "${CUDA_VER:-}" ]] && command -v nvcc &>/dev/null; then
        CUDA_VER=$(nvcc --version 2>/dev/null \
            | grep "release" | awk '{print $6}' | tr -d ',' || echo "")
    fi

    if [[ -z "${CUDA_VER:-}" ]]; then echo "cpu"; return; fi

    MAJOR=$(echo "${CUDA_VER}" | cut -d. -f1)
    MINOR=$(echo "${CUDA_VER}" | cut -d. -f2)

    if   [[ ${MAJOR} -ge 12 && ${MINOR} -ge 8 ]]; then echo "cu128"
    elif [[ ${MAJOR} -ge 12 && ${MINOR} -ge 6 ]]; then echo "cu126"
    elif [[ ${MAJOR} -ge 12 && ${MINOR} -ge 4 ]]; then echo "cu124"
    elif [[ ${MAJOR} -ge 12 && ${MINOR} -ge 1 ]]; then echo "cu121"
    elif [[ ${MAJOR} -ge 11 && ${MINOR} -ge 8 ]]; then echo "cu118"
    else echo "cpu"
    fi
}

# ── Install packages in strict dependency order ───────────────────────────────

echo ""
echo "========================================================"
echo " [1/9] numpy 1.23.5"
echo " Must be installed before TensorFlow to fix the"
echo " numpy.dtype ABI size mismatch (Expected 96 vs 88)."
echo "========================================================"
${PIP} install "numpy==1.23.5"

echo ""
echo "========================================================"
echo " [2/9] TensorFlow 2.13.0"
echo "========================================================"
${PIP} install "tensorflow==2.13.0"

echo ""
echo "========================================================"
echo " [3/9] Pre-pinning waymo SDK dependencies"
echo " waymo-open-dataset hardcodes old versions of these"
echo " packages. Installing them now at the required versions"
echo " before waymo runs its own dependency resolution."
echo "========================================================"
${PIP} install \
    "absl-py==1.4.0" \
    "google-auth==2.16.2" \
    "Pillow==9.2.0" \
    "scikit-learn==1.2.2" \
    "scipy>=1.7" \
    "cloudpickle>=1.3" \
    "decorator" \
    "attrs>=18.2.0"

echo ""
echo "========================================================"
echo " [4/9] tensorflow_probability 0.21.0"
echo "========================================================"
${PIP} install "tensorflow_probability==0.21.0"

echo ""
echo "========================================================"
echo " [5/9] typing_extensions 4.12.2"
echo " TF 2.13 metadata says <4.6 but torch 2.x and IPython"
echo " require >=4.10. TF works at runtime with 4.x — the"
echo " conflict is metadata-only. Install via wheel to bypass."
echo "========================================================"
${PIP} download "typing_extensions==4.12.2" -d /tmp/te_wheel --no-deps
${PIP} install /tmp/te_wheel/typing_extensions-4.12.2-py3-none-any.whl \
    --force-reinstall --no-deps
rm -rf /tmp/te_wheel

echo ""
echo "========================================================"
echo " [6/9] PyTorch"
TORCH_INDEX=$(detect_torch_index)
echo " CUDA build: ${TORCH_INDEX}"
if [[ "${TORCH_INDEX}" == "cpu" ]]; then
    echo ""
    echo " WARNING: No GPU/CUDA detected in this shell."
    echo " Login nodes have no GPU. If you know the cluster"
    echo " CUDA version, re-run with --cuda flag:"
    echo "   bash experiments/setup_env.sh --force --cuda cu121"
    echo ""
    echo " This cluster (Zaratan) has CUDA 12.3.0 -> cu121"
    echo " Check GPU nodes: srun --partition=gpu --gres=gpu:1"
    echo "                       --time=5:00 --pty bash -c"
    echo "                       'nvidia-smi | grep CUDA'"
fi
echo "========================================================"
if [[ "${TORCH_INDEX}" == "cpu" ]]; then
    ${PIP} install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cpu
else
    ${PIP} install torch torchvision torchaudio \
        --index-url "https://download.pytorch.org/whl/${TORCH_INDEX}"
fi

echo ""
echo "========================================================"
echo " [7/9] jaxlib 0.4.13 + jax 0.4.13"
echo " jaxlib 0.4.13 is NOT on PyPI. Must use JAX GCS bucket."
echo "========================================================"
${PIP} install "jaxlib==0.4.13" \
    -f https://storage.googleapis.com/jax-releases/jax_releases.html
${PIP} install "jax==0.4.13" \
    -f https://storage.googleapis.com/jax-releases/jax_releases.html

echo ""
echo "========================================================"
echo " [8/9] waymo-open-dataset SDK"
echo "========================================================"
${PIP} install "waymo-open-dataset-tf-2-12-0==1.6.7"

echo ""
echo "========================================================"
echo " [9/9] Remaining project dependencies"
echo "========================================================"
${PIP} install \
    "ml-dtypes==0.2.0" \
    "protobuf==3.20.3" \
    "pandas==1.5.3" \
    "matplotlib==3.6.1" \
    "pyyaml" \
    "tqdm" \
    "tensorboard" \
    "gcsfs" \
    "google-cloud-storage" \
    "grpcio" \
    "apache-beam" \
    "ipykernel" \
    "jupyterlab" \
    "ipywidgets" \
    "beautifulsoup4" \
    "filelock" \
    "toolz"

# ── Register Jupyter kernel ───────────────────────────────────────────────────
echo ""
echo "Registering Jupyter kernel..."
python -m ipykernel install --user \
    --name "${ENV_NAME}" \
    --display-name "Waymo TrajNet (Python 3.10)"

# Find the actual kernel.json location (may be in ~/.local or conda env share)
KERNEL_JSON=$(jupyter kernelspec list 2>/dev/null \
    | grep "${ENV_NAME}" \
    | awk '{print $2}')/kernel.json

echo "kernel.json location: ${KERNEL_JSON}"

if [[ -f "${KERNEL_JSON}" ]]; then
    python - <<PYEOF
import json
with open("${KERNEL_JSON}") as f:
    spec = json.load(f)
spec.setdefault("env", {})["PYTHONNOUSERSITE"] = "1"
with open("${KERNEL_JSON}", "w") as f:
    json.dump(spec, f, indent=1)
print("  PYTHONNOUSERSITE=1 written to kernel.json")
print("  argv[0]:", spec["argv"][0])
PYEOF
else
    echo "  WARNING: kernel.json not found at ${KERNEL_JSON}"
fi

# ── Final verification ────────────────────────────────────────────────────────
echo ""
echo "======================================================"
echo " Verifying installation..."
echo "======================================================"

python - <<'PYEOF'
import sys, importlib

print(f"Python: {sys.executable}  {sys.version.split()[0]}")
print()

all_ok = True
checks = [
    ("numpy",                  "1.23"),
    ("tensorflow",             "2.13"),
    ("tensorflow_probability", "0.21"),
    ("torch",                  None),
    ("typing_extensions",      None),
    ("jaxlib",                 "0.4.13"),
    ("absl",                   "1.4"),
    ("waymo metrics ops",      None),
]

for mod, expected in checks:
    try:
        if mod == "waymo metrics ops":
            from waymo_open_dataset.metrics.ops import py_metrics_ops
            from waymo_open_dataset.protos import motion_metrics_pb2
            print(f"  OK    waymo metrics ops")
            continue
        m   = importlib.import_module(mod)
        ver = getattr(m, "__version__", "installed")
        ok  = True
        if expected and not str(ver).startswith(expected):
            ok = False
        status = "OK   " if ok else "WARN "
        print(f"  {status} {mod} {ver}")
        if not ok:
            all_ok = False
    except Exception as e:
        print(f"  FAIL  {mod}: {e}")
        all_ok = False

print()
import torch
print(f"  torch CUDA available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        vram = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}  ({vram:.1f} GB)")
else:
    print("  (CUDA not available on login node — expected)")
    print("  CUDA will be available when running on a GPU compute node.")

print()
if all_ok:
    print("All checks passed.")
else:
    print("Some checks failed — see WARN/FAIL lines above.")
PYEOF

echo ""
echo "======================================================"
echo " Setup complete!"
echo ""
echo " To activate manually:"
echo "   source ${CONDA_BASE}/etc/profile.d/conda.sh"
echo "   conda activate ${ENV_NAME}"
echo ""
echo " If PyTorch was installed as CPU-only, reinstall with:"
echo "   conda activate ${ENV_NAME}"
echo "   pip install torch torchvision torchaudio \\"
echo "       --index-url https://download.pytorch.org/whl/cu121"
echo "======================================================"
