# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Main orchestration script for benchmark sweeps.

This script is called from within the sbatch job and coordinates:
1. Starting head node infrastructure (NATS, etcd)
2. Starting backend workers (prefill/decode/agg)
3. Starting frontends and nginx
4. Running benchmarks
5. Cleanup
"""

import argparse
import functools
import json
import logging
import os
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

from srtctl.cli.mixins import BenchmarkStageMixin, FrontendStageMixin, PostProcessStageMixin, WorkerStageMixin
from srtctl.core.config import load_config
from srtctl.core.health import wait_for_port
from srtctl.core.lockfile import write_lockfile
from srtctl.core.processes import (
    ManagedProcess,
    ProcessRegistry,
    setup_signal_handlers,
    start_process_monitor,
)
from srtctl.core.runtime import RuntimeContext
from srtctl.core.schema import SrtConfig
from srtctl.core.slurm import get_slurm_job_id, start_srun_process
from srtctl.core.status import JobStage, JobStatus, StatusReporter
from srtctl.core.topology import Endpoint, Process
from srtctl.logging_utils import setup_logging

logger = logging.getLogger(__name__)


@dataclass
class SweepOrchestrator(WorkerStageMixin, FrontendStageMixin, BenchmarkStageMixin, PostProcessStageMixin):
    """Main orchestrator for benchmark sweeps.

    Usage:
        config = load_config(config_path)  # Returns typed SrtConfig
        runtime = RuntimeContext.from_config(config, job_id)
        orchestrator = SweepOrchestrator(config, runtime)
        exit_code = orchestrator.run()
    """

    config: SrtConfig
    runtime: RuntimeContext

    @property
    def backend(self):
        """Access the backend config (implements BackendProtocol)."""
        return self.config.backend

    @functools.cached_property
    def endpoints(self) -> list[Endpoint]:
        """Compute endpoint allocation topology (cached).

        This is the single source of truth for endpoint assignments.
        """
        r = self.config.resources
        return self.backend.allocate_endpoints(
            num_prefill=r.num_prefill,
            num_decode=r.num_decode,
            num_agg=r.num_agg,
            gpus_per_prefill=r.gpus_per_prefill,
            gpus_per_decode=r.gpus_per_decode,
            gpus_per_agg=r.gpus_per_agg,
            gpus_per_node=r.gpus_per_node,
            available_nodes=self.runtime.nodes.worker,
        )

    @functools.cached_property
    def backend_processes(self) -> list[Process]:
        """Compute physical process topology from endpoints (cached)."""
        return self.backend.endpoints_to_processes(self.endpoints)

    def start_head_infrastructure(self, registry: ProcessRegistry) -> ManagedProcess:
        """Start NATS and etcd on the infra node.

        When etcd_nats_dedicated_node is enabled, services run on a dedicated node.
        Otherwise, they run on the head node (default behavior).
        """
        infra_node = self.runtime.nodes.infra
        logger.info("Starting infrastructure services (NATS, etcd)")
        logger.info("Infra node: %s", infra_node)

        setup_script = Path(__file__).parent / "setup_head.py"
        if not setup_script.exists():
            raise RuntimeError(f"setup_head.py not found at {setup_script}")

        setup_script_container = Path("/tmp/setup_head.py")
        infra_log = self.runtime.log_dir / "infra.out"

        cmd = [
            "python3",
            str(setup_script_container),
            "--name",
            self.config.name,
            "--log-dir",
            str(self.runtime.log_dir),
        ]
        if self.config.infra.nats_max_payload_mb is not None:
            cmd += ["--nats-max-payload-mb", str(self.config.infra.nats_max_payload_mb)]

        mounts = dict(self.runtime.container_mounts)
        mounts[setup_script] = setup_script_container
        # Mount host /tmp to container /host-tmp for etcd/nats data on local storage
        # This ensures etcd WAL writes go to fast local disk, not network storage
        mounts[Path("/tmp")] = Path("/host-tmp")

        proc = start_srun_process(
            command=cmd,
            nodelist=[infra_node],
            output=str(infra_log),
            container_image=str(self.runtime.container_image),
            container_mounts=mounts,
        )

        managed = ManagedProcess(
            name="infra_services",
            popen=proc,
            log_file=infra_log,
            node=infra_node,
            critical=True,
        )

        # 300s timeout to handle slow container imports on first run
        logger.info("Waiting for NATS (port 4222) on %s...", infra_node)
        if not wait_for_port(infra_node, 4222, timeout=300):
            raise RuntimeError("NATS failed to start")
        logger.info("NATS is ready")

        logger.info("Waiting for etcd (port 2379) on %s...", infra_node)
        if not wait_for_port(infra_node, 2379, timeout=300):
            raise RuntimeError("etcd failed to start")
        logger.info("etcd is ready")

        return managed

    def _replace_runtime_placeholders(self, value: str) -> str:
        """Replace supported runtime placeholders without interpreting other braces."""
        placeholders = {
            "infra_node": self.runtime.nodes.infra,
            "infra_node_ip": self.runtime.infra_node_ip,
            "head_node": self.runtime.nodes.head,
            "head_node_ip": self.runtime.head_node_ip,
            "job_id": self.runtime.job_id,
            "run_name": self.runtime.run_name,
            "log_dir": str(self.runtime.log_dir),
            "container_image": str(self.runtime.container_image),
        }
        result = value
        for name, replacement in placeholders.items():
            result = result.replace(f"{{{name}}}", str(replacement))
        return result

    def start_kvbm_hub(self, registry: ProcessRegistry) -> ManagedProcess:
        """Start the KVBM v2 hub service on the infra node."""
        del registry  # Process registration is handled by the caller after readiness.

        hub = self.config.kvbm_hub
        if not hub.enabled:
            raise RuntimeError("KVBM hub is not enabled")

        infra_node = self.runtime.nodes.infra
        hub_log = self.runtime.log_dir / "kvbm_hub.out"

        logger.info("Starting KVBM hub on %s", infra_node)
        cmd = [
            "kvbm_hub",
            "--discovery-port",
            str(hub.discovery_port),
            "--control-port",
            str(hub.control_port),
            "--features",
            ",".join(hub.features),
            "--block-size",
            str(hub.block_size),
            "--max-seq-len",
            str(hub.max_seq_len),
            "--layout",
            hub.layout,
        ]
        if hub.velo_port is not None:
            cmd.extend(["--velo-port", str(hub.velo_port)])
        if hub.g2_memory is not None:
            cmd.extend(["--g2-memory", str(hub.g2_memory)])
        if hub.g2_block is not None:
            cmd.extend(["--g2-block", str(hub.g2_block)])
        if hub.bind_addr:
            cmd.extend(["--bind-addr", self._replace_runtime_placeholders(hub.bind_addr)])
        if hub.kv_index_advertise_host:
            cmd.extend(["--kv-index-advertise-host", self._replace_runtime_placeholders(hub.kv_index_advertise_host)])
        if hub.kv_index_zmq_bind:
            cmd.extend(["--kv-index-zmq-bind", self._replace_runtime_placeholders(hub.kv_index_zmq_bind)])
        if hub.kvbm_config:
            cmd.extend(["--kvbm-config", self._replace_runtime_placeholders(hub.kvbm_config)])
        if hub.kvbm:
            cmd.extend(["--kvbm", self._replace_runtime_placeholders(hub.kvbm)])

        proc = start_srun_process(
            command=cmd,
            nodelist=[infra_node],
            output=str(hub_log),
            container_image=str(self.runtime.container_image),
            container_mounts=dict(self.runtime.container_mounts),
            env_to_set=dict(hub.environment) or None,
        )

        managed = ManagedProcess(
            name="kvbm_hub",
            popen=proc,
            log_file=hub_log,
            node=infra_node,
            critical=True,
        )

        logger.info("Waiting for KVBM hub control port %s on %s...", hub.control_port, infra_node)
        if not wait_for_port(infra_node, hub.control_port, timeout=hub.startup_timeout_seconds):
            raise RuntimeError("KVBM hub failed to start")
        logger.info("KVBM hub is ready")

        return managed

    def _write_mooncake_config(self) -> Path:
        """Write the Mooncake JSON config consumed by vLLM workers."""
        hub = self.config.mooncake_hub
        config_path = self.runtime.log_dir / hub.config_filename
        master_server_address = f"{self.runtime.infra_node_ip}:{hub.rpc_port}"

        mooncake_config = {
            "mode": self._replace_runtime_placeholders(hub.mode),
            "metadata_server": self._replace_runtime_placeholders(hub.metadata_server),
            "master_server_address": master_server_address,
            "global_segment_size": hub.global_segment_size,
            "local_buffer_size": hub.local_buffer_size,
            "protocol": self._replace_runtime_placeholders(hub.protocol),
            "device_name": self._replace_runtime_placeholders(hub.device_name),
            "enable_offload": hub.enable_offload,
        }
        if hub.config_extra:
            mooncake_config.update(
                {
                    self._replace_runtime_placeholders(str(key)): self._replace_runtime_placeholders(str(value))
                    if isinstance(value, str)
                    else value
                    for key, value in hub.config_extra.items()
                }
            )

        config_path.write_text(json.dumps(mooncake_config, indent=2, sort_keys=True) + "\n")
        logger.info("Wrote Mooncake config: %s", config_path)
        return config_path

    def _check_mooncake_rdma_prereqs(self) -> None:
        """Fail early when Mooncake RDMA cannot register GPU KV-cache memory."""
        hub = self.config.mooncake_hub
        protocol = self._replace_runtime_placeholders(hub.protocol).lower()
        if protocol != "rdma":
            return

        if os.environ.get("SRTCTL_SKIP_MOONCAKE_GDR_CHECK"):
            logger.warning("Skipping Mooncake RDMA GPUDirect preflight because SRTCTL_SKIP_MOONCAKE_GDR_CHECK is set")
            return

        peermem_value = str(hub.environment.get("WITH_NVIDIA_PEERMEM", os.environ.get("WITH_NVIDIA_PEERMEM", "0")))
        use_legacy_peermem = peermem_value.upper() in {"1", "ON", "TRUE", "YES"}
        if use_legacy_peermem:
            check_script = (
                'if [ -d /sys/module/nvidia_peermem ] || [ -d /sys/module/nv_peer_mem ]; then '
                'exit 0; '
                'fi; '
                'echo "$(hostname): missing nvidia_peermem/nv_peer_mem"; '
                "exit 1"
            )
            check_description = "legacy nvidia-peermem GPUDirect prerequisites"
        else:
            check_script = (
                'ok=1; '
                'if [ ! -e /proc/driver/nvidia/version ]; then '
                'echo "$(hostname): missing NVIDIA driver"; ok=0; '
                'fi; '
                'if ! ls /dev/infiniband/uverbs* >/dev/null 2>&1; then '
                'echo "$(hostname): missing RDMA verbs devices"; ok=0; '
                'fi; '
                "exit $((1-ok))"
            )
            check_description = "DMA-BUF GPUDirect prerequisites"
        nodes = tuple(dict.fromkeys(self.runtime.nodes.worker or (self.runtime.nodes.infra,)))
        slurm_job_id = get_slurm_job_id()

        if slurm_job_id and nodes:
            cmd = [
                "srun",
                "--jobid",
                slurm_job_id,
                "--overlap",
                "--nodes",
                str(len(nodes)),
                "--ntasks",
                str(len(nodes)),
                "--ntasks-per-node",
                "1",
                "--nodelist",
                ",".join(nodes),
                "bash",
                "-lc",
                check_script,
            ]
        else:
            cmd = ["bash", "-lc", check_script]

        logger.info("Checking Mooncake RDMA %s on worker nodes", check_description)
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode == 0:
            if not use_legacy_peermem:
                logger.info(
                    "Mooncake RDMA will use CUDA DMA-BUF path; set WITH_NVIDIA_PEERMEM=1 to require legacy peermem"
                )
            return

        details = "\n".join(part for part in (result.stdout.strip(), result.stderr.strip()) if part)
        if not details:
            details = f"preflight command exited with {result.returncode}"
        if use_legacy_peermem:
            raise RuntimeError(
                "Mooncake RDMA was configured with WITH_NVIDIA_PEERMEM=1, but "
                "nvidia_peermem or nv_peer_mem is not loaded on every worker node. "
                "Either ask the cluster admins to load the module, or set WITH_NVIDIA_PEERMEM=0 "
                f"to use CUDA DMA-BUF. Details:\n{details}"
            )
        raise RuntimeError(
            "Mooncake RDMA DMA-BUF preflight failed before vLLM startup. "
            "Expected NVIDIA driver access and RDMA verbs devices on every worker node. "
            "If those are present but Mooncake still fails later, validate CUDA DMA-BUF directly "
            f"with ib_write_bw --use_cuda_dmabuf. Details:\n{details}"
        )

    def start_mooncake_hub(self, registry: ProcessRegistry) -> ManagedProcess:
        """Start the Mooncake Store master service on the infra node."""
        del registry  # Process registration is handled by the caller after readiness.

        hub = self.config.mooncake_hub
        if not hub.enabled:
            raise RuntimeError("Mooncake hub is not enabled")

        infra_node = self.runtime.nodes.infra
        hub_log = self.runtime.log_dir / "mooncake_hub.out"
        self._check_mooncake_rdma_prereqs()
        self._write_mooncake_config()

        logger.info("Starting Mooncake hub on %s", infra_node)
        cmd = [
            "mooncake_master",
            "--rpc_port",
            str(hub.rpc_port),
            "--rpc_address",
            self._replace_runtime_placeholders(hub.rpc_address),
        ]
        if hub.enable_http_metadata_server:
            cmd.extend(
                [
                    "--enable_http_metadata_server=true",
                    "--http_metadata_server_host",
                    self._replace_runtime_placeholders(hub.http_metadata_server_host),
                    "--http_metadata_server_port",
                    str(hub.http_metadata_server_port),
                ]
            )
        if hub.enable_offload:
            cmd.append("--enable_offload=true")

        proc = start_srun_process(
            command=cmd,
            nodelist=[infra_node],
            output=str(hub_log),
            container_image=str(self.runtime.container_image),
            container_mounts=dict(self.runtime.container_mounts),
            env_to_set=dict(hub.environment) or None,
        )

        managed = ManagedProcess(
            name="mooncake_hub",
            popen=proc,
            log_file=hub_log,
            node=infra_node,
            critical=True,
        )

        logger.info("Waiting for Mooncake hub RPC port %s on %s...", hub.rpc_port, infra_node)
        if not wait_for_port(infra_node, hub.rpc_port, timeout=hub.startup_timeout_seconds):
            raise RuntimeError("Mooncake hub failed to start")
        logger.info("Mooncake hub is ready")

        return managed

    def _print_connection_info(self) -> None:
        """Print srun commands for connecting to nodes."""
        container_args = f"--container-image={self.runtime.container_image}"
        mounts_str = ",".join(f"{src}:{dst}" for src, dst in self.runtime.container_mounts.items())
        if mounts_str:
            container_args += f" --container-mounts={mounts_str}"

        logger.info("")
        logger.info("=" * 60)
        logger.info("Connection Commands")
        logger.info("=" * 60)
        logger.info("Frontend URL: http://%s:8000", self.runtime.nodes.head)
        logger.info("")
        logger.info("To connect to head node (%s):", self.runtime.nodes.head)
        logger.info(
            "  srun %s --jobid %s -w %s --overlap --pty bash",
            container_args,
            self.runtime.job_id,
            self.runtime.nodes.head,
        )

        # Print worker node connection commands
        for node in self.runtime.nodes.worker:
            if node != self.runtime.nodes.head:
                logger.info("")
                logger.info("To connect to worker node (%s):", node)
                logger.info(
                    "  srun %s --jobid %s -w %s --overlap --pty bash",
                    container_args,
                    self.runtime.job_id,
                    node,
                )

        logger.info("=" * 60)
        logger.info("")

    def run(self) -> int:
        """Run the complete sweep."""
        # Create status reporter (fire-and-forget, no-op if not configured)
        reporter = StatusReporter.from_config(self.config.reporting, self.runtime.job_id)
        reporter.report_started(self.config, self.runtime)

        logger.info("Sweep Orchestrator")
        logger.info("Job ID: %s", self.runtime.job_id)
        logger.info("Run name: %s", self.runtime.run_name)
        logger.info("Config: %s", self.config.name)
        logger.info("Infra node: %s", self.runtime.nodes.infra)
        logger.info("Head node: %s", self.runtime.nodes.head)
        logger.info("Worker nodes: %s", ", ".join(self.runtime.nodes.worker))
        if self.config.profiling.enabled:
            logger.info("Profiling: %s", self.config.profiling.type)

        # Write initial lockfile with config + SLURM context (fingerprint added after run)
        write_lockfile(self.runtime.log_dir.parent, self.config)

        registry = ProcessRegistry(job_id=self.runtime.job_id)
        stop_event = threading.Event()
        setup_signal_handlers(stop_event, registry)
        start_process_monitor(stop_event, registry)

        exit_code = 1

        try:
            # Stage 1: Head infrastructure (NATS, etcd)
            reporter.report(JobStatus.STARTING, JobStage.HEAD_INFRASTRUCTURE, "Starting head infrastructure")
            head_proc = self.start_head_infrastructure(registry)
            registry.add_process(head_proc)

            if self.config.kvbm_hub.enabled:
                reporter.report(JobStatus.STARTING, JobStage.HEAD_INFRASTRUCTURE, "Starting KVBM hub")
                kvbm_hub_proc = self.start_kvbm_hub(registry)
                registry.add_process(kvbm_hub_proc)
            if self.config.mooncake_hub.enabled:
                reporter.report(JobStatus.STARTING, JobStage.HEAD_INFRASTRUCTURE, "Starting Mooncake hub")
                mooncake_hub_proc = self.start_mooncake_hub(registry)
                registry.add_process(mooncake_hub_proc)

            # Stage 2: Workers
            reporter.report(JobStatus.WORKERS, JobStage.WORKERS, "Starting workers")
            worker_procs = self.start_all_workers()
            registry.add_processes(worker_procs)

            # Stage 3: Frontend
            reporter.report(JobStatus.FRONTEND, JobStage.FRONTEND, "Starting frontend")
            frontend_procs = self.start_frontend(registry)
            for proc in frontend_procs:
                registry.add_process(proc)

            self._print_connection_info()

            # Stage 4: Benchmark (status reported AFTER health check passes)
            exit_code = self.run_benchmark(registry, stop_event, reporter)

        except Exception as e:
            logger.exception("Error during sweep: %s", e)
            reporter.report(JobStatus.FAILED, JobStage.CLEANUP, str(e))
            exit_code = 1

        finally:
            logger.info("Cleanup")
            reporter.report_completed(exit_code)
            stop_event.set()
            registry.cleanup()
            if exit_code != 0:
                registry.print_failure_details()
            # Run post-processing (AI analysis if enabled)
            self.run_postprocess(exit_code)

        return exit_code


def main():
    """Main entry point."""
    from dataclasses import replace

    parser = argparse.ArgumentParser(description="Run benchmark sweep")
    parser.add_argument("config", type=str, help="Path to YAML configuration file")
    args = parser.parse_args()

    setup_logging()

    try:
        config_path = Path(args.config)
        if not config_path.exists():
            logger.error("Config file not found: %s", config_path)
            sys.exit(1)

        config = load_config(config_path)

        # Check for setup_script override from CLI (passed via env var)
        setup_script_override = os.environ.get("SRTCTL_SETUP_SCRIPT")
        if setup_script_override:
            logger.info("Setup script override: %s", setup_script_override)
            config = replace(config, setup_script=setup_script_override)

        job_id = get_slurm_job_id()
        if not job_id:
            logger.error("Not running in SLURM (SLURM_JOB_ID not set)")
            sys.exit(1)

        # Type narrowing: job_id is str after the check above
        assert job_id is not None
        runtime = RuntimeContext.from_config(config, job_id)
        orchestrator = SweepOrchestrator(config=config, runtime=runtime)
        exit_code = orchestrator.run()

        sys.exit(exit_code)

    except Exception as e:
        logger.exception("Fatal error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
