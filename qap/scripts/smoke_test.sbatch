#!/bin/bash
# Slurm wrapper for qap/scripts/smoke_test.sh.
# Submit FROM THE REPOSITORY ROOT (the folder that contains qap/ and .venv/):
#     sbatch qap/scripts/smoke_test.sbatch
# Log: smoke_<jobid>.out in the repository root.  Watch it with:  tail -f smoke_<jobid>.out
#
#SBATCH --job-name=qap_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=smoke_%j.out
#SBATCH --error=smoke_%j.out
# Uncomment / edit if your cluster needs them:
##SBATCH --partition=<gpu-partition>
##SBATCH --account=<account>

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit with sbatch from the repository root}"

# environment: only the project venv (no conda base on top of it)
if command -v conda >/dev/null 2>&1; then conda deactivate 2>/dev/null || true; fi
source .venv/bin/activate

# assets must resolve to ~/.maniskill/data (textures are hard-coded there)
unset MS_ASSET_DIR

export PYTHONPATH=$PWD:${PYTHONPATH:-}
export QAP_DATA_ROOT=${QAP_DATA_ROOT:-$PWD/qap_runs_smoke}
export OMP_NUM_THREADS=1          # many simulator processes; avoid thread oversubscription
export PYTHONUNBUFFERED=1         # stream progress into the log

echo "host: $(hostname)  job: ${SLURM_JOB_ID:-}  cpus: ${SLURM_CPUS_PER_TASK:-?}  gpu: ${CUDA_VISIBLE_DEVICES:-none}"
echo "python: $(which python)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import numpy, scipy, zarr, torch; print('numpy', numpy.__version__, 'scipy', scipy.__version__, 'zarr', zarr.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"

start=$(date +%s)
bash qap/scripts/smoke_test.sh
echo "smoke test wall time: $(( ($(date +%s) - start) / 60 )) min"
