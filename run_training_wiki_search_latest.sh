#!/bin/bash

#SBATCH -J tap_jupyter
#SBATCH -o log_wiki_after_fix.out
#SBATCH -e log_wiki_after_fix.err
#SBATCH -p gh-dev                          # Adjust partition as needed
#SBATCH --nodes=2                      # 2 nodes for 2 GPUs (1 per node)
#SBATCH --ntasks-per-node=1
#SBATCH --time=2:00:00

# ------------------------------------------------------------------
# Helpful SLURM variables
# ------------------------------------------------------------------
# $SLURM_PROCID   : global rank   (0 … WORLD_SIZE-1)
# $SLURM_NODEID   : node  rank    (0 … NODES-1)
# $SLURM_JOB_NODELIST : host list
# ------------------------------------------------------------------

# Load modules
module load gcc/13.2.0
module load cuda/12.8
export CC=gcc
export CXX=g++

# Load environment
deactivate  # Clear any existing environments
source /scratch/09143/arnabd/newproj/.venv/bin/activate  # Adjust path as needed

# Configuration
export MODEL_NAME="Qwen/Qwen3-4B-Instruct-2507"
export ENV_ID="wiki-search"
export RUN_NAME="wiki-search"
verifiers_root=$(python3 -c "import os, verifiers; print(os.path.dirname(os.path.dirname(verifiers.__file__)))")
config_file="/scratch/09143/arnabd/newproj3/verifiers_mybranch/verifiers/configs/local/vf-rl/wiki-search.toml"


# WandB Configuration
export WANDB_PROJECT="wiki-search"
export WANDB_NAME="wiki-search-1.5b"
export WANDB_API_KEY="2f7ab7a7c411fd9ad1ccddb815acdcd2ebc5c732"  # Replace with your actual API key
export WANDB_MODE="online"  # Use "offline" if you want offline mode
export WANDB_DIR="/scratch/09143/arnabd/newproj/wandb/"  # Optional: specify wandb directory

# Weave Configuration
export WANDB_ENABLE_WEAVE="true"  # Enable Weave tracing
export WANDB_WEAVE_PROJECT="$WANDB_PROJECT"  # Use same project as WandB

# These libs are for deepspeed CPU offloading (if needed)
# CUDA + NVIDIA HPC SDK math libs
export NVHPC_HOME=/home1/apps/nvidia/Linux_aarch64/25.3
export CUDA_HOME=/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8
export CUDA_LIB64="$CUDA_HOME/lib64"
export MATH_LIBS_LIB64="$NVHPC_HOME/math_libs/12.8/lib64"

export LD_LIBRARY_PATH="$MATH_LIBS_LIB64:$CUDA_LIB64:$CUDA_LIB64/stubs:${LD_LIBRARY_PATH}"
export LIBRARY_PATH="$MATH_LIBS_LIB64:$CUDA_LIB64:$CUDA_LIB64/stubs:${LIBRARY_PATH}"
export CPATH="$CUDA_HOME/include:${CPATH}"

# Get master node hostname for vLLM server
export VLLM_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export VLLM_PORT=8000
export VLLM_BASE_URL="http://${VLLM_ADDR}:${VLLM_PORT}/v1"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "VLLM_ADDR=$VLLM_ADDR"
echo "VLLM_BASE_URL=$VLLM_BASE_URL"

# Create a temporary config file with the vLLM server host/port
#temp_config=$(mktemp /tmp/training_config_XXXXXX.toml)
#cp "$config_file" "$temp_config"
# Create a temporary config file with the vLLM server host/port in the same directory
config_dir=$(dirname "$config_file")
config_basename=$(basename "$config_file" .toml)
job_uuid=$(python3 -c "import uuid; print(uuid.uuid4())" 2>/dev/null || uuidgen 2>/dev/null || echo "$$-$(date +%s)")
temp_config="${config_dir}/${config_basename}_${job_uuid}.toml"
cp "$config_file" "$temp_config"
echo "Created temporary config: $temp_config"

# Add vllm_server_host and vllm_server_port to trainer.args using Python
python3 << EOF
try:
    import tomllib
    import tomli_w
except ImportError:
    # Fallback: use simple string replacement if tomli-w not available
    import re
    with open("$temp_config", "r") as f:
        content = f.read()
    
    # Check if trainer.args section exists
    if "[trainer.args]" in content:
        # Append vllm_server_host and port if not already present
        if "vllm_server_host" not in content:
            content = content.rstrip() + "\nvllm_server_host = \"$VLLM_ADDR\"\nvllm_server_port = $VLLM_PORT\n"
    else:
        # Add trainer.args section if it doesn't exist
        content = content.rstrip() + "\n\n[trainer.args]\nvllm_server_host = \"$VLLM_ADDR\"\nvllm_server_port = $VLLM_PORT\n"
    
    with open("$temp_config", "w") as f:
        f.write(content)
else:
    # Use proper TOML parsing if available
    with open("$config_file", "rb") as f:
        config = tomllib.load(f)
    
    # Ensure trainer.args exists
    if "trainer" not in config:
        config["trainer"] = {}
    if "args" not in config["trainer"]:
        config["trainer"]["args"] = {}
    
    # Set the vLLM server host and port
    config["trainer"]["args"]["vllm_server_host"] = "$VLLM_ADDR"
    config["trainer"]["args"]["vllm_server_port"] = $VLLM_PORT
    
    # Write the modified config
    with open("$temp_config", "wb") as f:
        tomli_w.dump(config, f)
EOF

# Start vLLM server on the master node (node 0, GPU 0)
srun --export=ALL -N 1 -n 1 -w "$VLLM_ADDR" bash -lc '
    echo "[$(hostname)] starting vllm server on GPU-0"
    CUDA_VISIBLE_DEVICES=0 HF_HOME=$WORK/.huggingface \
        vf-vllm --model "$MODEL_NAME" \
            --enable-auto-tool-choice \
            --tool-call-parser hermes \
            --gpu-memory-utilization 0.80 \
            --enforce-eager
' &

# Wait a bit for vLLM server to start
sleep 10

# Start training on the second node (node 1, GPU 0)
# Start training on the second node (node 1, GPU 0)
srun --export=ALL -N 1 -n 1 --exclude "$VLLM_ADDR" bash -lc "
    echo \"[\$(hostname)] starting training\"
    source /scratch/09143/arnabd/newproj/.venv/bin/activate
    CUDA_VISIBLE_DEVICES=0 HF_HOME=\$WORK/.huggingface TRITON_CACHE_DIR=/tmp/.triton_cache \\
        vf-install \"$ENV_ID\" && \\
        vf-train @ \"$temp_config\"
" &

# Wait for both processes
wait

#rm -rf "$temp_config"