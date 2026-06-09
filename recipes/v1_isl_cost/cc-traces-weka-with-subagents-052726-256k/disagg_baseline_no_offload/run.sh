#!/bin/bash
# Run all disagg baselines WITHOUT offloading
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for config in "$SCRIPT_DIR"/*.yaml; do
  echo "Submitting: $config"
  srtctl submit -f "$config"
done
