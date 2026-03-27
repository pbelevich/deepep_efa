#!/bin/bash

# DeepEP-EFA test script for Slurm
# Tests intranode first (1 node), then internode (2 nodes)

#SBATCH --job-name=deepep-efa
#SBATCH --nodes=2
#SBATCH --ntasks-per-node 1
#SBATCH --exclusive
#SBATCH --wait-all-nodes=1

set -ex;

## Set libfabric flags to use EFA
export FI_PROVIDER=efa
export FI_EFA_USE_DEVICE_RDMA=1
export FI_EFA_FORK_SAFE=1

## Set this flag for debugging EFA
# export FI_LOG_LEVEL=warn

# NCCL settings (for torch.distributed init)
export NCCL_BUFFSIZE=8388608
export NCCL_P2P_NET_CHUNKSIZE=1048576

# Get hostname info
mpirun -N 1 bash -c 'echo $(hostname): $(cat /sys/devices/virtual/dmi/id/board_asset_tag | tr -d " ")'

export MASTER_PORT=29500
export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
export NUM_NODES=${SLURM_NNODES}
export NUM_GPUS=8

# Test 1: Intranode test (runs on single node, 8 GPUs)
echo "=== Running intranode test ==="
srun -l --nodes=1 --ntasks=1 \
    --container-image ./deepep-efa.sqsh \
    --mpi=pmix --cpu-bind=none \
    bash -c 'cd /app/deepep_efa && \
        WORLD_SIZE=1 RANK=0 MASTER_ADDR=localhost MASTER_PORT=29500 \
        python3 tests/test_intranode.py'

# Test 2: Internode test (runs on 2 nodes, 8 GPUs each = 16 GPUs)
echo "=== Running internode test ==="
srun -l \
    --container-image ./deepep-efa.sqsh \
    --mpi=pmix --cpu-bind=none \
    bash -c 'cd /app/deepep_efa && \
        WORLD_SIZE=${NUM_NODES} RANK=${SLURM_NODEID} \
        MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} \
        python3 tests/test_internode.py'

echo "=== All tests complete ==="
