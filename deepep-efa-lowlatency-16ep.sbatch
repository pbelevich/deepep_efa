#!/bin/bash

# DeepEP-EFA low-latency test for 16EP (2 nodes, 8 GPUs each = 16 GPUs)

#SBATCH --job-name=deepep-ll16
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --output=slurm-deepep-ll16-%j.out
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

# Low-latency test at 16EP — same parameters as 8EP for fair comparison
# 128 tokens, hidden=7168, topk=8, experts=288
echo "=== Running LL 16EP test ==="
srun -l \
    --container-image ./deepep-efa.sqsh \
    --mpi=pmix --cpu-bind=none \
    bash -c 'cd /app/deepep_efa && \
        WORLD_SIZE='"${NUM_NODES}"' RANK=${SLURM_NODEID} \
        MASTER_ADDR='"${MASTER_ADDR}"' MASTER_PORT='"${MASTER_PORT}"' \
        DEEPEP_PROFILE=1 \
        DEEPEP_LL_PIPELINE=0 \
        DEEPEP_COOP=1 \
        python3 tests/test_low_latency.py --num-tokens 128 --hidden 7168 --num-topk 8 --num-experts 288'

echo "=== LL 16EP test complete ==="
