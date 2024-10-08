#!/bin/bash

#SBATCH --job-name=energy_locality_vs_globality
#SBATCH --output=benchmark.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1                  # 1 tarefa (processo) por nó
#SBATCH --cpus-per-task=8           # 8 CPUs por tarefa
#SBATCH --gres=gpu:1                # 1 GPU por nó
#SBATCH --time=7-00:00:00           # Limite de tempo para execução
#SBATCH --partition=gpuPartition    # Partição padrão (ajustar conforme necessário)

USER="rfg@av.it.pt"
SCRIPT_NAME="benchmark_log.py"

# Activate virtual environment and run the script
PYTHONPATH=/slurm_shared/home/${USER}/electronics_ts_forecasting /slurm_shared/home/${USER}/electronics_ts_forecasting/venv/bin/python /slurm_shared/home/${USER}/electronics_ts_forecasting/${SCRIPT_NAME}