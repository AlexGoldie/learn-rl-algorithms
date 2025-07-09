#!/bin/bash

# Check if GPU names are provided
if [ -z "$1" ]; then
  echo "Please provide GPU names (e.g., 0,1,2)"
  exit 1
fi

# GPU names passed as the first argument
GPU_NAMES=$1

# Run the Docker command with the specified GPUs.
docker run -it --rm --gpus '"device='$GPU_NAMES'"' -v $(pwd):/learn-rl-algorithms -w /learn-rl-algorithms --env-file .env meta-learning