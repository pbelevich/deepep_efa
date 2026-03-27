#!/bin/bash

# DeepEP-EFA low-latency test (1 node, 8 GPUs)

#SBATCH --job-name=deepep-ll
#SBATCH --nodes=1
#SBATCH --ntasks-per-node 1
#SBATCH --exclusive
#SBATCH --output=slurm-deepep-ll-%j.out

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

# Low-latency test (1 node, 8 GPUs, uses torch.multiprocessing.spawn internally)
echo "=== Running low-latency test ==="
srun -l \
    --container-image ./deepep-efa.sqsh \
    --mpi=pmix --cpu-bind=none \
    bash -c 'cd /app/deepep_efa && \
        WORLD_SIZE=1 RANK=0 MASTER_ADDR=localhost MASTER_PORT=29500 \
        DEEPEP_PROFILE=1 DEEPEP_PROFILE_INTERVAL=20 DEEPEP_PROFILE_WARMUP=10 \
        python3 tests/test_low_latency.py --num-tokens 128 --hidden 7168 --num-topk 8 --num-experts 288'

echo "=== Low-latency test complete ==="
