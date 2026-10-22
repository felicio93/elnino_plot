#!/bin/bash --login
# One-off: mesh -> regular-grid interpolation weights.
#SBATCH -D {{WORKDIR}}
#SBATCH -J elnino_prep
#SBATCH -A {{ACCOUNT}}
#SBATCH -p {{PARTITION}}
#SBATCH -N 1 -n 1
#SBATCH --mem={{MEM}}
#SBATCH -t {{WALLTIME}}
#SBATCH -o {{LOGDIR}}/prepare.out
#SBATCH -e {{LOGDIR}}/prepare.err

set -e
export PYTHONPATH={{REPO}}:$PYTHONPATH
echo "=== elnino prepare ==="
{{PY}} -m elnino --config {{CONFIG}} prepare
echo "=== done ==="
