# Analysis Tools Agent Guide

This guide explains how to use the analysis tools in this directory for benchmarking LLM inference configurations.

## Directory Structure

Job outputs are stored in `/lustre/fsw/coreai_dlfw_dev/karenc/srt-slurm/outputs/`.

Each job directory follows the naming convention: `{job_id}_{config_name}`

Example: `2188558_agg_tep4x6_eagle_kv_routing_concurrency12`

Key files in each job directory:
- `logs/config.yaml` - The configuration used for the run
- `logs/profile_export_aiperf.json` - Benchmark metrics (if run succeeded)
- `logs/*.out` - Worker output logs
- `logs/sweep_*.log` - Main orchestration log

---

## Quick Commands Reference

```bash
# Extract metrics as CSV (vLLM backend)
python analysis/david_viz.py --dir <job_ids> --vllm --header --csv

# Extract metrics with conditional prefill columns
python analysis/david_viz.py --dir <job_ids> --vllm --condp --header --csv

# Create Pareto plot
uv run python analysis/plot_pareto.py \
  --series "config1" --dir <ids1> \
  --series "config2" --dir <ids2> \
  --output analysis/plots/my_pareto.png

# Monitor running jobs
squeue -u $USER
tail -f outputs/<job_id>*/logs/*.out

# Submit new job
uv run srtctl apply -f path/to/config.yaml

# Cancel jobs
scancel <job_id>
```

---

## Renaming Job Directories

When a job directory is just the job ID (e.g., `2188558`), rename it to include the config name.

### Batch Rename Script
```bash
cd /lustre/fsw/coreai_dlfw_dev/karenc/srt-slurm/outputs
for id in 2188558 2188559 2188560; do
  if [ -d "$id" ]; then
    config_file="$id/logs/config.yaml"
    [ ! -f "$config_file" ] && config_file="$id/config.yaml"
    name=$(grep "^name:" "$config_file" 2>/dev/null | head -1 | sed 's/name: *//')
    if [ -n "$name" ]; then
      echo "$id -> ${id}_${name}"
      mv "$id" "${id}_${name}"
    fi
  fi
done
```

### Marking Failed Jobs
For jobs that failed (no `profile_export_aiperf.json`), prefix with `failed_`:
```bash
mv outputs/2166900_config_name outputs/2166900_failed_config_name
```

---

## david_viz.py - Benchmark Results Extraction

Extracts metrics from job directories and outputs TSV/CSV for analysis.

### Basic Usage

```bash
cd /lustre/fsw/coreai_dlfw_dev/karenc/srt-slurm

# IMPORTANT: Must specify --trtllm or --vllm depending on backend
python analysis/david_viz.py --dir <job_ids> --trtllm [options]
python analysis/david_viz.py --dir <job_ids> --vllm [options]
```

### Required Flag: Backend Selection

| Flag | Use When |
|------|----------|
| `--trtllm` | Jobs using TensorRT-LLM backend |
| `--vllm` | Jobs using vLLM backend |

### Common Options

| Option | Description |
|--------|-------------|
| `--dir <ids>` | Space-separated job IDs |
| `--header` | Include header row |
| `--csv` | Output comma-separated (default is tab-separated) |
| `--condp` | Include conditional prefill policy columns (`condp_policy`, `max_num_tokens`) |

### Examples

**Single job (TRT-LLM):**
```bash
python analysis/david_viz.py --dir 2166917 --trtllm --header --csv
```

**Multiple jobs (vLLM):**
```bash
python analysis/david_viz.py --dir 2188558 2188559 2188560 --vllm --header --csv
```

**With conditional prefill columns:**
```bash
python analysis/david_viz.py --dir 2207244 2207245 2207246 --vllm --condp --header --csv
```

### Output Columns

Standard columns:
- `dataset`, `srtslurm_id`, `config_name`, `concurrency`
- `request_count`, `errors [pct]`, `runtime_error`
- `ttft_avg_ms`, `ttft_p50_ms`, `ttft_p99_ms`
- `itl_avg_ms`, `itl_p50_ms`, `itl_p99_ms`
- `output_tput_per_user`, `total_token_tput`, `total_token_tput_per_gpu`, `request_tput`
- `goodput [ttft_sla/itl_sla]`
- `kv_total_blocks`, `kv_blocksize`, `kv_total_workspace_GiB`
- `kv_util_max`, `kv_reused_blocks`, `kv_missed_blocks`, `kv_hit_rate`

With `--condp`, adds after `config_name`:
- `condp_policy` (e.g., `isl_bounding | 2048 / 0.8`)
- `max_num_tokens` (extracts `max-num-batched-tokens` for vLLM, `max_num_tokens` for TRT-LLM)

### Key Metrics Explained

| Metric | Description |
|--------|-------------|
| `output_tput_per_user` | Output tokens per second per user (tok/user) - latency metric |
| `total_token_tput_per_gpu` | Total output tokens per second per GPU (tok/gpu) - efficiency metric |
| `ttft_p50_ms` | Time to first token, 50th percentile (ms) |
| `itl_p50_ms` | Inter-token latency, 50th percentile (ms) |

### Failed Jobs

Jobs without `profile_export_aiperf.json` (failed runs) will still output a row with:
- Config columns populated
- `runtime_error` showing the detected error from `error_summary` field
- Metric columns empty

---

## plot_pareto.py - Pareto Frontier Plots

Creates throughput vs latency Pareto plots comparing multiple configurations.

### Basic Usage

```bash
cd /lustre/fsw/coreai_dlfw_dev/karenc/srt-slurm

uv run python analysis/plot_pareto.py \
  --series "<series_name>" --dir <job_ids> \
  --series "<series_name>" --dir <job_ids> \
  [options]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `--series <name>` | Name for this series (appears in legend) |
| `--dir <ids>` | Job IDs for this series (space-separated) |
| `--output <path>` | Output file path (default: `pareto.png`) |
| `--title <text>` | Plot title |
| `--label-points` | Add concurrency and TTFT labels to each point |
| `--no-frontier` | Don't draw Pareto frontier lines |
| `--vllm` | Use vLLM backend for parsing |
| `--trtllm` | Use TRT-LLM backend for parsing |
| `--json-output <path>` | Export series data to JSON for later modification |
| `--json-input <path>` | Load series data from JSON instead of reading job dirs |

### Plot Axes

- **X-axis**: Output Tokens/s/User (latency metric - higher is better)
- **Y-axis**: Total Output Tokens/s/GPU (throughput metric - higher is better)

### Examples

**Two series comparison:**
```bash
uv run python analysis/plot_pareto.py \
  --series "agg_tep4x6" --dir 2191491 2191492 2191493 2191494 2191495 \
  --series "agg_tep2x6" --dir 2191496 2191497 2191498 2191499 2191500 \
  --label-points \
  --title "vLLM Qwen3-235B Aggregated Comparison" \
  --output analysis/plots/my_pareto.png \
  --vllm
```

**Export to JSON for later editing:**
```bash
uv run python analysis/plot_pareto.py \
  --series "config_a" --dir 2191491 2191492 2191493 \
  --series "config_b" --dir 2191496 2191497 2191498 \
  --output analysis/plots/my_pareto.png \
  --json-output analysis/plots/my_pareto.json \
  --vllm
```

**Load from JSON (after manual editing):**
```bash
uv run python analysis/plot_pareto.py \
  --json-input analysis/plots/my_pareto.json \
  --output analysis/plots/my_pareto_modified.png
```

### JSON Export Format

The JSON file contains series data that can be manually edited:
```json
{
  "series_name": {
    "points": [
      {"x": 1.23, "y": 45.6, "concurrency": 8, "ttft": 123.4, "job_id": "2191491"},
      ...
    ],
    "gpu_count": 48
  }
}
```

### Grouping Jobs into Series

Group jobs by their configuration type (e.g., topology, model, backend):

1. List job directories to see config names:
```bash
ls -d outputs/219149*
```

2. Group by common prefix (e.g., `agg_tep4x6`, `disagg_tep2x3p_tep2x4d`)

3. Each series should vary only by concurrency for a proper Pareto curve

---

## Monitoring Running Jobs

### Check Running Jobs
```bash
squeue -u $USER
```

### Watch Job Logs for Errors
```bash
# Check latest output from a specific job
tail -100 outputs/<job_id>*/logs/*.out

# Watch for errors/warnings in real-time
tail -f outputs/<job_id>*/logs/sweep_*.log | grep -i "error\|warning\|fail"

# Check if benchmark is progressing
tail -f outputs/<job_id>*/logs/*.out | grep -i "concurrency\|request\|benchmark"
```

### Common Error Patterns to Look For
- `CUDA out of memory` / `OOM` - GPU memory exhausted
- `NCCL error` - Communication failure between GPUs
- `AssertionError` - Code assertion failed
- `Timeout` - Operation took too long
- `Connection refused` - Service not started

### Check aiperf Error Summary
The `profile_export_aiperf.json` file contains an `error_summary` field:
```bash
jq '.error_summary' outputs/<job_id>*/logs/profile_export_aiperf.json
```

---

## Job Management

### Cancel Jobs
```bash
# Cancel single job
scancel <job_id>

# Cancel multiple jobs
scancel 2207244 2207245 2207246

# Cancel all your jobs
scancel -u $USER
```

### Delete Job Directories
```bash
# Delete single directory
rm -rf outputs/<job_id>_*

# Delete range of job directories
for id in $(seq 2207244 2207251); do
  rm -rf outputs/${id}*
done
```

### Submit New Jobs
```bash
# Submit a single config
uv run srtctl apply -f recipes/path/to/config.yaml

# Submit all configs in a directory
for config in recipes/my_experiment/*.yaml; do
  echo "Submitting: $config"
  uv run srtctl apply -f "$config"
done
```

---

## Creating New Recipe Configs

### Key Config Parameters (vLLM)

```yaml
name: my_experiment_concurrency8
resources:
  gpu_type: "b200"
  prefill_nodes: 2
  prefill_workers: 2
  decode_nodes: 4
  decode_workers: 4
  gpus_per_prefill: 8
  gpus_per_decode: 4

backend:
  type: vllm
  vllm_config:
    prefill:
      gpu-memory-utilization: 0.90
      max-num-batched-tokens: 256
      kv-transfer-config: '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'
    decode:
      gpu-memory-utilization: 0.90
      max-num-batched-tokens: 256
      kv-transfer-config: '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'
```

### KV Transfer Config Options

**Without offloading (NixlConnector only):**
```yaml
kv-transfer-config: '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'
```

**With offloading (MultiConnector with OffloadingConnector):**
```yaml
kv-transfer-config: '{"kv_connector":"MultiConnector","kv_connectors":[{"kv_connector":"NixlConnector","kv_role":"kv_both"},{"kv_connector":"OffloadingConnector","kv_role":"kv_both"}]}'
```

### Conditional Prefill Parameters

Add to vLLM config for conditional prefill:
```yaml
enable-conditional-disagg: true
conditional-disagg-policy: isl_bounding
conditional-disagg-cutoff-isl: 4096
conditional-disagg-long-ratio: 0.8
max-num-batched-tokens: 4352  # cutoff + buffer
```

---

## Workflow Examples

### 1. New jobs completed, need analysis

```bash
# Check what jobs exist
ls outputs/220*

# Rename if needed
cd /lustre/fsw/coreai_dlfw_dev/karenc/srt-slurm/outputs
for id in 2202387 2202388 2202389; do
  if [ -d "$id" ]; then
    config_file="$id/logs/config.yaml"
    [ ! -f "$config_file" ] && config_file="$id/config.yaml"
    name=$(grep "^name:" "$config_file" 2>/dev/null | head -1 | sed 's/name: *//')
    if [ -n "$name" ]; then
      mv "$id" "${id}_${name}"
    fi
  fi
done

# Extract metrics
cd /lustre/fsw/coreai_dlfw_dev/karenc/srt-slurm
python analysis/david_viz.py --dir 2202387 2202388 2202389 --vllm --header --csv

# Create pareto plot
uv run python analysis/plot_pareto.py \
  --series "my_config" --dir 2202387 2202388 2202389 \
  --label-points \
  --output analysis/plots/my_analysis.png \
  --vllm
```

### 2. Compare offload vs no-offload configurations

```bash
# Extract metrics for both
python analysis/david_viz.py --dir 2202387 2202388 2202389 --vllm --csv --header > no_offload.csv
python analysis/david_viz.py --dir 2192595 2192596 2192597 --vllm --csv --header > with_offload.csv

# Create comparison Pareto
uv run python analysis/plot_pareto.py \
  --series "No Offload 3P+3D" --dir 2202387 2202388 2202389 \
  --series "With Offload 3P+3D" --dir 2192595 2192596 2192597 \
  --label-points \
  --title "Offload vs No-Offload Comparison" \
  --output analysis/plots/offload_comparison.png \
  --json-output analysis/plots/offload_comparison.json \
  --vllm
```

### 3. Monitor and debug failing jobs

```bash
# Check which jobs are running
squeue -u $USER

# Watch for errors
tail -f outputs/220*_*/logs/*.out 2>/dev/null | grep -i "error\|fail\|oom"

# Check specific job's error summary
for dir in outputs/2207*_*/; do
  json_file=$(find "$dir" -name "profile_export_aiperf.json" 2>/dev/null | head -1)
  if [ -n "$json_file" ]; then
    echo "=== $dir ==="
    jq '.error_summary' "$json_file"
  fi
done
```

### 4. Create configs for new experiment

```bash
# Copy existing config as template
cp recipes/baseline/config.yaml recipes/my_experiment/new_config.yaml

# Edit parameters (e.g., change kv-transfer-config, workers, concurrency)
# Then submit
uv run srtctl apply -f recipes/my_experiment/new_config.yaml
```

---

## Performance Comparison Table Format

When comparing configurations, use this table format:

| Config | GPUs | tok/user range | tok/gpu range | TTFT p50 range |
|--------|------|----------------|---------------|----------------|
| 3P+3D (TP2) no-offload | 48 | [1.1, 1.2, 1.7, 7.6, 28.5] | [46.7, 45.7, 44.4, 42.6, 37.6] | [97, 175, 343, 1256, 4534] |
| 3P+3D (TP2) offload | 48 | [1.2, 1.5, 3.1, 15.4, 56.9] | [43.2, 43.0, 40.9, 35.2, 25.3] | [166, 247, 534, 2459, 9143] |

Values are listed in order of increasing concurrency (e.g., conc 8, 12, 24, 64, 128).

---

## Tips

1. **Always use `--vllm` or `--trtllm`** - The scripts need to know which backend to parse

2. **Rename directories immediately** - Makes it easier to identify configs later

3. **Export JSON for complex plots** - Allows manual editing of series names, colors, point selection

4. **Check error_summary first** - Faster than reading through large log files

5. **Group by concurrency for Pareto** - Each series should have runs at different concurrency levels

6. **Use `uv run` for scripts requiring dependencies** - Ensures correct Python environment
