#!/bin/bash

# Get current user info
USER_NAME=$(id -un)
USER_ID=$(id -u)
GROUP_ID=$(id -g)

# Build the Docker image with current user info
docker build \
  --build-arg USERNAME="${USER_NAME}" \
  --build-arg USER_UID="${USER_ID}" \
  --build-arg USER_GID="${GROUP_ID}" \
  -t meta-learning \
  .