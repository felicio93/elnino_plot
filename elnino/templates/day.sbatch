#!/bin/bash --login
# One array task per day: regrid model, load satellite, write stats + frames.
#SBATCH -D {{WORKDIR}}
#SBATCH -J elnino_day
#SBATCH -A {{ACCOUNT}}
#SBATCH -p {{PARTITION}}
#SBATCH --array=1-{{NTASKS}}%{{THROTTLE}}
#SBATCH -N 1 -n 1
#SBATCH --mem={{MEM}}
#SBATCH -t {{WALLTIME}}
#SBATCH -o {{LOGDIR}}/day_%a.out
#SBATCH -e {{LOGDIR}}/day_%a.err

set -e
export PYTHONPATH={{REPO}}:$PYTHONPATH
DAY=$(sed -n "${SLURM_ARRAY_TASK_ID}p" {{MANIFEST}})
echo "=== elnino day ${DAY} (task ${SLURM_ARRAY_TASK_ID}) ==="
{{PY}} -m elnino --config {{CONFIG}} day --date "${DAY}"
echo "=== done ${DAY} ==="
