# Analyzing Benchmark Results

This guide covers all aspects of analyzing your benchmark results, from launching the interactive dashboard to programmatically parsing raw data.

## Table of Contents

1. [Overview](#overview)
2. [Understanding Output Structure](#understanding-output-structure)
3. [Interactive Dashboard](#interactive-dashboard)
4. [Command-Line Analysis](#command-line-analysis)
5. [Metrics Deep Dive](#metrics-deep-dive)
6. [Comparing Experiments](#comparing-experiments)
7. [Exporting Data](#exporting-data)
8. [Troubleshooting Analysis](#troubleshooting-analysis)

---

## Overview

The analysis toolkit provides two primary ways to analyze benchmark results:

| Tool | Use Case | Access |
|------|----------|--------|
| **Interactive Dashboard** | Visual exploration, comparing runs, real-time filtering | `make dashboard` |
| **Programmatic Analysis** | Automation, custom analysis, CI/CD integration | Python API or raw JSON |

### When to Use Each Tool

- **Dashboard**: Ideal for exploring results, comparing configurations, identifying trends, and presenting findings to stakeholders
- **Command-Line/Python**: Best for automated analysis pipelines, custom metrics, and integration with other tools

---

## Understanding Output Structure

### Directory Layout

Each benchmark run creates a directory under `logs/` (or your configured output directory):

```
logs/
  3667_1P_4D_20251110_192145/           # Job ID + Topology + Timestamp
    3667.json                            # Run metadata (required)
    sa-bench_isl_1024_osl_1024/          # Profiler results directory
      concurrency_16.json                # Results for each concurrency level
      concurrency_32.json
      concurrency_64.json
      ...
    watchtower-navy-cn01_prefill_w0.err  # Worker stderr logs
    watchtower-navy-cn01_prefill_w0.out  # Worker stdout logs
    watchtower-navy-cn02_decode_w0.err
    watchtower-navy-cn02_decode_w0.out
    watchtower-navy-cn01_prefill_config.json  # Node configuration snapshots
    watchtower-navy-cn02_decode_config.json
    .cache/                              # Parquet cache files (auto-generated)
      benchmark_results.parquet
      node_metrics.parquet
```

### Directory Naming Convention

The run directory name encodes key information:

```
{SLURM_JOB_ID}_{TOPOLOGY}_{TIMESTAMP}
      |            |           |
      |            |           +-- YYYYMMDD_HHMMSS
      |            +-- 1P_4D (1 prefill, 4 decode) or 8A (8 aggregated)
      +-- SLURM job identifier
```

### Metadata File (`{jobid}.json`)

The JSON metadata file is the source of truth for run configuration:

```json
{
  "run_metadata": {
    "slurm_job_id": "3667",
    "run_date": "20251110_192145",
    "container": "ghcr.io/sgl-project/sglang:v0.4.1-cu121",
    "prefill_nodes": 1,
    "decode_nodes": 4,
    "prefill_workers": 1,
    "decode_workers": 4,
    "gpus_per_node": 8,
    "gpu_type": "NVIDIA H100 80GB HBM3",
    "mode": "disaggregated",
    "model_dir": "/models/DeepSeek-V3"
  },
  "profiler_metadata": {
    "type": "sa-bench",
    "isl": "1024",
    "osl": "1024",
    "concurrencies": "16x32x64x128x256"
  },
  "tags": ["baseline", "h100"]
}
```

### Benchmark Result Files

Each concurrency level produces a JSON file in the profiler results directory:

```json
{
  "max_concurrency": 64,
  "output_throughput": 15234.5,
  "total_token_throughput": 23456.7,
  "request_throughput": 14.8,
  "mean_ttft_ms": 245.3,
  "mean_tpot_ms": 32.1,
  "mean_itl_ms": 31.8,
  "mean_e2el_ms": 1456.2,
  "median_ttft_ms": 198.4,
  "median_tpot_ms": 28.9,
  "p99_ttft_ms": 892.1,
  "p99_tpot_ms": 78.4,
  "total_input_tokens": 65536,
  "total_output_tokens": 65536,
  "completed": 1000,
  "duration": 67.5
}
```

### Log Files

Worker log files contain runtime metrics:

- **`.err` files**: Application logs with batch-level metrics
- **`.out` files**: Standard output (often empty or contains startup info)

Example log line format:
```
[2025-11-04 05:31:43 DP0 TP0 EP0] Prefill batch, #new-seq: 18, #new-token: 16384,
#cached-token: 0, token usage: 0.00, #running-req: 0, #queue-req: 0,
#prealloc-req: 0, #inflight-req: 0, input throughput (token/s): 0.00
```

### Configuration Snapshots

`*_config.json` files capture the complete node configuration at runtime:

- GPU information (count, type, memory, driver version)
- Server arguments (TP/DP/PP size, attention backend, KV cache settings)
- Environment variables (NCCL, CUDA, SGLANG settings)
- Command-line arguments actually passed

---

## Interactive Dashboard

### Launching the Dashboard

```bash
uv run streamlit run analysis/dashboard/app.py
```

The dashboard opens at `http://localhost:8501` by default.

### Dashboard Configuration

On the left sidebar, you will see:

1. **Logs Directory Path**: Set the path to your outputs directory (defaults to `outputs/`)

### Run Selection

The sidebar provides powerful filtering options:

#### GPU Type Filter
Filter runs by GPU hardware (e.g., H100, A100, L40S). Useful when comparing across different hardware generations.

#### Topology Filter
Filter by worker configuration:
- Disaggregated: `1P/4D`, `2P/8D`, etc.
- Aggregated: `4A`, `8A`, etc.

#### ISL/OSL Filter
Filter by input/output sequence length combinations (e.g., `1024/1024`, `2048/512`).

#### Container Filter
Filter by container image version to compare software updates.

#### Tags Filter
Filter by custom tags you have assigned to runs. Tags help organize experiments:
- `baseline` - Control runs
- `optimized` - Runs with optimizations
- `production` - Production-ready configurations

### Dashboard Tabs

The dashboard has five main tabs:

#### 1. Pareto Graph Tab

**Purpose**: Visualize the efficiency trade-off between throughput per GPU and throughput per user.

**What You See**:
- X-axis: **Output TPS/User** - Token generation rate experienced by each user (1000/TPOT)
- Y-axis: **Output TPS/GPU** or **Total TPS/GPU** - GPU utilization efficiency

**Key Features**:
- **Y-axis toggle**: Switch between Output TPS/GPU (decode tokens only) and Total TPS/GPU (input + output)
- **TPS/User cutoff line**: Add a vertical line to mark your target throughput requirement
- **Pareto Frontier**: Highlight the efficient frontier where no other configuration is strictly better

**Interpreting the Graph**:
- Points **up and to the right** are better (higher efficiency AND higher per-user throughput)
- Points on the **Pareto frontier** represent optimal trade-offs
- Use the cutoff line to identify configurations meeting your latency requirements

**Metric Calculations**:

$$\text{Output TPS/GPU} = \frac{\text{Total Output Throughput (tokens/s)}}{\text{Total Number of GPUs}}$$

$$\text{Output TPS/User} = \frac{1000}{\text{Mean TPOT (ms)}}$$

**Data Export**: Click "Download Data as CSV" to export all data points.

#### 2. Latency Analysis Tab

**Purpose**: Analyze latency metrics across concurrency levels.

**Graphs Displayed**:

1. **TTFT (Time to First Token)**: Time from request submission to first output token
   - Critical for perceived responsiveness
   - Should remain stable under load

2. **TPOT (Time Per Output Token)**: Average time between consecutive output tokens
   - Determines streaming speed
   - Lower TPOT = faster generation

3. **ITL (Inter-Token Latency)**: Similar to TPOT but may include queueing delays
   - Useful for diagnosing scheduling issues

**Summary Statistics**: Table showing min/max values for each metric across selected runs.

#### 3. Node Metrics Tab

**Purpose**: Deep dive into runtime behavior of individual workers.

**Aggregation Modes**:
- **Individual nodes**: See every worker separately
- **Group by DP rank**: Average metrics across tensor parallel workers within each data parallel group
- **Aggregate all nodes**: Single averaged line per run

**Prefill Node Metrics**:
- **Input Throughput**: Tokens/s being processed in prefill
- **Inflight Requests**: Requests sent to decode workers awaiting completion
- **KV Cache Utilization**: Memory pressure indicator
- **Queued Requests**: Backpressure indicator

**Decode Node Metrics**:
- **Running Requests**: Active generation requests
- **Generation Throughput**: Output tokens/s
- **KV Cache Utilization**: Memory pressure
- **Queued Requests**: Decode capacity indicator

**Disaggregation Metrics** (Stacked or Separate views):
- **Prealloc Queue**: Requests waiting for memory allocation
- **Transfer Queue**: Requests waiting for KV cache transfer
- **Running**: Requests actively generating

#### 4. Rate Match Tab

**Purpose**: Verify prefill/decode capacity balance.

**Interpretation**:
- **Lines should align**: System is balanced
- **Decode consistently below prefill**: Need more decode nodes
- **Decode above prefill**: Prefill is the bottleneck, decode underutilized

**Toggle**: Convert from tokens/s to requests/s using ISL/OSL for clearer comparison.

**Note**: This tab only applies to disaggregated runs (prefill/decode split). Aggregated runs are skipped.

#### 5. Configuration Tab

**Purpose**: Review the exact configuration of each run.

**Information Displayed**:
- **Overview**: Node count, GPU type, ISL/OSL, profiler type
- **Topology**: Physical node assignments, service distribution
- **Node Config**: Command-line arguments for each worker
- **Environment**: Environment variables by category (NCCL, SGLANG, CUDA, etc.)

### Managing Tags

Tags help organize and filter your experiments:

1. **Adding Tags**: Expand a run in the sidebar Tags section, type a tag name, click "Add"
2. **Removing Tags**: Click the "x" button next to any existing tag
3. **Filtering by Tags**: Use the Tags filter in the Filters section

Tags are stored in the run's `{jobid}.json` file and persist across sessions.

---

## Command-Line Analysis

### Accessing Raw JSON Results

Browse directly to the profiler results:

```bash
# List all runs
ls logs/

# View a specific run's metadata
cat logs/3667_1P_4D_20251110_192145/3667.json | jq .

# List benchmark results
ls logs/3667_1P_4D_20251110_192145/sa-bench_isl_1024_osl_1024/

# View results for specific concurrency
cat logs/3667_1P_4D_20251110_192145/sa-bench_isl_1024_osl_1024/concurrency_64.json | jq .
```

### Using jq for Analysis

Extract specific metrics across concurrency levels:

```bash
# Get throughput for all concurrency levels in a run
for f in logs/3667_*/sa-bench_*/concurrency_*.json; do
  echo "$(basename $f): $(jq '.output_throughput' $f) TPS"
done

# Extract mean TTFT across runs
jq -r '[.max_concurrency, .mean_ttft_ms] | @tsv' logs/*/sa-bench_*/concurrency_*.json

# Find the best throughput across all runs
find logs -name "concurrency_*.json" -exec jq -r \
  '[input_filename, .output_throughput] | @tsv' {} \; | \
  sort -t$'\t' -k2 -nr | head -10
```

### Python API

For programmatic analysis, use the `RunLoader` class:

```python
from analysis.srtlog import RunLoader, NodeAnalyzer

# Load all runs from a directory
loader = RunLoader("logs")
runs = loader.load_all()

# Filter runs
h100_runs = [r for r in runs if "H100" in (r.metadata.gpu_type or "")]

# Access metadata
for run in runs:
    print(f"Job {run.job_id}: {run.metadata.topology_label}")
    print(f"  GPU: {run.metadata.gpu_type}")
    print(f"  Complete: {run.is_complete}")

    # Access benchmark results
    for i, concurrency in enumerate(run.profiler.concurrency_values):
        tps = run.profiler.output_tps[i]
        ttft = run.profiler.mean_ttft_ms[i]
        print(f"  C={concurrency}: {tps:.0f} TPS, TTFT={ttft:.1f}ms")

# Convert to DataFrame for analysis
df = loader.to_dataframe(runs)
print(df.describe())

# Analyze node-level metrics
analyzer = NodeAnalyzer()
nodes = analyzer.parse_run_logs("logs/3667_1P_4D_20251110_192145")
prefill_nodes = analyzer.get_prefill_nodes(nodes)
decode_nodes = analyzer.get_decode_nodes(nodes)
```

### Pandas Analysis Examples

```python
import pandas as pd
from analysis.srtlog import RunLoader

loader = RunLoader("logs")
df = loader.to_dataframe()

# Best throughput per topology
best_by_topology = df.groupby(["Prefill Workers", "Decode Workers"])["Output TPS"].max()
print(best_by_topology)

# Average latency by concurrency
avg_latency = df.groupby("Concurrency")[["Mean TTFT (ms)", "Mean TPOT (ms)"]].mean()
print(avg_latency)

# Find optimal concurrency (best TPS/GPU while meeting latency target)
target_ttft = 500  # ms
valid_points = df[df["Mean TTFT (ms)"] <= target_ttft]
best = valid_points.loc[valid_points["Output TPS/GPU"].idxmax()]
print(f"Optimal: Concurrency={best['Concurrency']}, TPS/GPU={best['Output TPS/GPU']:.1f}")
```

---

## Metrics Deep Dive

### Throughput Metrics

| Metric | Description | Unit |
|--------|-------------|------|
| **Output TPS** | Total output tokens generated per second across all users | tokens/s |
| **Total TPS** | Total tokens processed (input + output) per second | tokens/s |
| **Request Throughput** | Number of requests completed per second | requests/s |
| **Request Goodput** | Successful requests per second (excludes errors) | requests/s |
| **Output TPS/GPU** | Output TPS divided by total GPU count | tokens/s/GPU |
| **Output TPS/User** | Per-user generation rate (1000/TPOT) | tokens/s |

### Latency Metrics

| Metric | Description | What It Tells You |
|--------|-------------|-------------------|
| **TTFT** | Time to First Token | User-perceived responsiveness |
| **TPOT** | Time Per Output Token | Streaming speed during generation |
| **ITL** | Inter-Token Latency | Token spacing (similar to TPOT) |
| **E2EL** | End-to-End Latency | Total request duration |

### Understanding Percentiles

- **Mean**: Average across all requests (sensitive to outliers)
- **Median (p50)**: Middle value (50% of requests faster, 50% slower)
- **p90**: 90% of requests complete faster than this
- **p99**: 99% of requests complete faster than this (tail latency)
- **Standard Deviation**: Spread around the mean

**Best Practices**:
- Use **p99** for SLA commitments
- Use **median** for typical user experience
- Large gap between median and p99 indicates scheduling issues or resource contention

### What "Good" Metrics Look Like

These are general guidelines; actual targets depend on your use case:

| Metric | Good | Acceptable | Concerning |
|--------|------|------------|------------|
| TTFT (p99) | < 500ms | 500-1000ms | > 1000ms |
| TPOT (mean) | < 30ms | 30-50ms | > 50ms |
| Output TPS/GPU | > 200 | 100-200 | < 100 |
| KV Cache Utilization | 40-80% | 20-90% | > 95% or < 10% |
| Queue Depth | 0-10 | 10-50 | > 50 (growing) |

**Note**: These vary significantly by:
- Model size (larger models = slower)
- Hardware (H100 vs A100 vs L40S)
- Sequence lengths (longer = slower)
- Batch sizes and concurrency

---

## Comparing Experiments

### Using Tags for Organization

Establish a tagging convention for your team:

```
# Example tags
baseline-v1          # First baseline measurement
optimized-chunked    # With chunked prefill
production-20251115  # Production configuration snapshot
regression-test      # Automated regression tests
```

### A/B Comparison Patterns

**Compare two configurations**:
1. Run both configurations with identical:
   - ISL/OSL settings
   - Concurrency levels
   - Hardware (if possible)
2. Tag runs appropriately (e.g., `configA`, `configB`)
3. In dashboard:
   - Filter to show only your tagged runs
   - Select both runs for side-by-side comparison
   - Use Pareto graph to see efficiency differences

**Identify regressions**:
```python
from analysis.srtlog import RunLoader

loader = RunLoader("logs")
runs = loader.load_all()

# Get baseline and current runs by tag
baseline = [r for r in runs if "baseline" in r.tags]
current = [r for r in runs if "current" in r.tags]

# Compare at same concurrency
for b, c in zip(baseline, current):
    for i, conc in enumerate(b.profiler.concurrency_values):
        if conc in c.profiler.concurrency_values:
            j = c.profiler.concurrency_values.index(conc)
            b_tps = b.profiler.output_tps[i]
            c_tps = c.profiler.output_tps[j]
            diff = (c_tps - b_tps) / b_tps * 100
            print(f"Concurrency {conc}: {diff:+.1f}% change")
```

### Filtering by Parameters

In the dashboard sidebar:
1. Use **Topology filter** to compare same worker ratios
2. Use **ISL/OSL filter** to compare same workload profiles
3. Use **Container filter** to compare software versions
4. Use **GPU Type filter** to compare hardware

---

## Exporting Data

### CSV Export

From the dashboard Pareto tab, click "Download Data as CSV" to export:
- All selected runs
- All concurrency levels
- All computed metrics (TPS, TPS/GPU, TPS/User, latencies)

### JSON Export

Raw JSON is already available in the logs directory. To consolidate:

```bash
# Export all results to a single JSON file
python -c "
import json
from analysis.srtlog import RunLoader

loader = RunLoader('logs')
df = loader.to_dataframe()
print(df.to_json(orient='records', indent=2))
" > all_results.json
```

### Parquet Export (Cached Data)

The analysis system automatically caches parsed data as Parquet files:

```python
import pandas as pd

# Read cached benchmark results
df = pd.read_parquet("logs/3667_1P_4D_20251110_192145/.cache/benchmark_results.parquet")

# Read cached node metrics
nodes_df = pd.read_parquet("logs/3667_1P_4D_20251110_192145/.cache/node_metrics.parquet")
```

### Integration with Other Tools

**Grafana/InfluxDB**:
```python
from influxdb_client import InfluxDBClient
from analysis.srtlog import RunLoader

loader = RunLoader("logs")
df = loader.to_dataframe()

# Write to InfluxDB
with InfluxDBClient(url="http://localhost:8086", token="...") as client:
    write_api = client.write_api()
    # Convert DataFrame to line protocol and write
```

**Jupyter Notebooks**:
```python
# In a Jupyter cell
from analysis.srtlog import RunLoader
import matplotlib.pyplot as plt

loader = RunLoader("logs")
df = loader.to_dataframe()

# Create custom visualizations
df.groupby("Concurrency")["Output TPS"].mean().plot(kind="bar")
plt.title("Average Throughput by Concurrency")
plt.show()
```

---

## Troubleshooting Analysis

### Dashboard Won't Load

**Symptoms**: Dashboard shows spinner indefinitely or errors on startup

**Solutions**:
1. Check logs directory exists: `ls -la logs/`
2. Verify at least one run has `{jobid}.json`: `ls logs/*/*.json`
3. Check for Python errors: `uv run streamlit run analysis/dashboard/app.py 2>&1`
4. Clear Streamlit cache: `rm -rf ~/.streamlit/cache`

### Missing Runs in Dashboard

**Symptoms**: Some runs don't appear in the run selector

**Causes and Solutions**:

1. **No metadata file**: Each run must have `{jobid}.json`
   ```bash
   # Check if metadata exists
   ls logs/3667_*/3667.json
   ```

2. **No benchmark results**: Runs without profiler output are skipped
   ```bash
   # Check for profiler results
   ls logs/3667_*/sa-bench_*/
   ```

3. **Profiling jobs**: `torch-profiler` type runs are intentionally skipped
   ```bash
   # Check profiler type
   jq '.profiler_metadata.type' logs/3667_*/3667.json
   ```

4. **Cache invalidation**: Force reload by clicking "Sync Now" or restarting dashboard

### Incomplete Run Warning

**Symptoms**: Dashboard shows "Job X is incomplete - Missing concurrencies: [128, 256]"

**Causes**:
- Benchmark timed out before completing all concurrency levels
- Job was cancelled mid-run
- Profiler crashed at higher concurrencies

**Solutions**:
1. Check SLURM logs for timeout or OOM errors
2. Re-run with longer timeout
3. Reduce max concurrency for resource-constrained setups

### No Node Metrics Found

**Symptoms**: Node Metrics tab shows "No log files found"

**Causes**:
- Log files don't match expected pattern
- Logs were not captured (stderr redirect issue)

**Solutions**:
1. Verify log file naming: `ls logs/3667_*/*_prefill_*.err`
2. Check file contents: `head logs/3667_*/*_prefill_*.err`
3. Verify log format matches expected patterns (see Log Files section)

### Slow Dashboard Loading

**Symptoms**: Dashboard takes a long time to load or refresh

**Causes**:
- Many runs to parse
- Cache invalidation
- Large log files

**Solutions**:
1. Parquet caching speeds up subsequent loads automatically
2. Delete old runs you no longer need
3. Use filters to reduce the number of selected runs
4. Increase `_cache_version` in `components.py` only when parser changes

### Incorrect Metrics

**Symptoms**: Metrics don't match expected values or show as "N/A"

**Causes**:
- Benchmark output format changed
- Incomplete benchmark run
- Parse error in result files

**Solutions**:
1. Verify raw JSON is valid:
   ```bash
   jq . logs/3667_*/sa-bench_*/concurrency_64.json
   ```

2. Check for required fields:
   ```bash
   jq 'keys' logs/3667_*/sa-bench_*/concurrency_64.json
   ```

3. Clear parquet cache and reload:
   ```bash
   rm -rf logs/3667_*/.cache/
   ```

---

## Quick Reference

### Launch Dashboard
```bash
make dashboard
# or
uv run streamlit run analysis/dashboard/app.py
```

### Python API Quick Start
```python
from analysis.srtlog import RunLoader, NodeAnalyzer

# Load runs
loader = RunLoader("logs")
runs = loader.load_all()

# Get DataFrame
df = loader.to_dataframe()

# Parse node logs
analyzer = NodeAnalyzer()
nodes = analyzer.parse_run_logs("logs/3667_1P_4D_20251110_192145")
```

### Key File Locations
- Run metadata: `logs/{run_dir}/{jobid}.json`
- Benchmark results: `logs/{run_dir}/{profiler}_isl_{isl}_osl_{osl}/concurrency_*.json`
- Worker logs: `logs/{run_dir}/*_{prefill|decode}_*.err`
- Node configs: `logs/{run_dir}/*_config.json`
- Cache files: `logs/{run_dir}/.cache/*.parquet`
