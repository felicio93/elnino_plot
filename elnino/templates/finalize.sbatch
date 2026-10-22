#!/bin/bash --login
# After all days: summary figures (task 1) and one GIF+MP4 per plot type.
#SBATCH -D {{WORKDIR}}
#SBATCH -J elnino_final
#SBATCH -A {{ACCOUNT}}
#SBATCH -p {{PARTITION}}
#SBATCH --array=1-{{NTASKS}}
#SBATCH -N 1 -n 1
#SBATCH --mem={{MEM}}
#SBATCH -t {{WALLTIME}}
#SBATCH -o {{LOGDIR}}/finalize_%a.out
#SBATCH -e {{LOGDIR}}/finalize_%a.err

set -e
export PYTHONPATH={{REPO}}:$PYTHONPATH
TASKS=({{TASKS}})
TASK=${TASKS[$((SLURM_ARRAY_TASK_ID-1))]}
echo "=== elnino finalize: ${TASK} ==="
{{PY}} -m elnino --config {{CONFIG}} finalize --task "${TASK}"
echo "=== done ${TASK} ==="
