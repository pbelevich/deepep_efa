#!/bin/bash

# DeepEP-EFA intranode test (1 node, 8 GPUs)

#SBATCH --job-name=deepep-intra
#SBATCH --nodes=1
#SBATCH --ntasks-per-node 1
#SBATCH --exclusive
#SBATCH --output=slurm-deepep-intra-%j.out

set -ex;

## Set libfabric flags to use EFA
export FI_PROVIDER=efa
export FI_EFA_USE_DEVICE_RDMA=1
export FI_EFA_FORK_SAFE=1

# NCCL settings (for torch.distributed init)
export NCCL_BUFFSIZE=8388608
export NCCL_P2P_NET_CHUNKSIZE=1048576

# Get hostname info
echo "Host: $(hostname)"

export MASTER_PORT=29500
export MASTER_ADDR=localhost

# Intranode test (1 node, 8 GPUs, uses torch.multiprocessing.spawn internally)
echo "=== Running intranode test ==="
srun -l \
    --container-image ./deepep-efa.sqsh \
    --mpi=pmix --cpu-bind=none \
    bash -c 'cd /app/deepep_efa && \
        WORLD_SIZE=1 RANK=0 MASTER_ADDR=localhost MASTER_PORT=29500 \
        python3 tests/test_intranode.py'

echo "=== Intranode test complete ==="
