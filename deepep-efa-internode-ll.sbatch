#!/bin/bash

# DeepEP-EFA internode test with low-latency compatibility (2 nodes, 8 GPUs each = 16 GPUs)

#SBATCH --job-name=deepep-interll
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --output=slurm-deepep-interll-%j.out
#SBATCH --wait-all-nodes=1

set -ex;

## Set libfabric flags to use EFA
export FI_PROVIDER=efa
export FI_EFA_USE_DEVICE_RDMA=1
export FI_EFA_FORK_SAFE=1

# NCCL settings (for torch.distributed init)
export NCCL_BUFFSIZE=8388608
export NCCL_P2P_NET_CHUNKSIZE=1048576

# Get hostname info
echo "Hosts:"
srun -l hostname

export MASTER_PORT=29500
export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
export NUM_NODES=${SLURM_NNODES}

echo "MASTER_ADDR=${MASTER_ADDR}, NUM_NODES=${NUM_NODES}"

# Internode test with low-latency compatibility
echo "=== Running internode test with --test-ll-compatibility ==="
srun -l \
    --container-image ./deepep-efa.sqsh \
    --mpi=pmix --cpu-bind=none \
    bash -c 'cd /app/deepep_efa && \
        WORLD_SIZE='"${NUM_NODES}"' RANK=${SLURM_NODEID} \
        MASTER_ADDR='"${MASTER_ADDR}"' MASTER_PORT='"${MASTER_PORT}"' \
        python3 tests/test_internode.py --test-ll-compatibility'

echo "=== Internode test with LL compatibility complete ==="
