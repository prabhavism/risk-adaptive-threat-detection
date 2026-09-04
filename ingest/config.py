"""
Centralized ingestion configuration. Everything tunable about capture,
flow timeouts, and memory bounds lives here rather than scattered
through ingest/*.py -- see docs/interfaces.md and ingest/README.md for
what each value does and why.
"""
from dataclasses import dataclass


@dataclass
class IngestConfig:
    # --- Capture source ---
    interface: str = "eth0"
    pcap_path: str = "data/demo.pcap"
    replay_speed: float = 1.0          # 1.0 = real-time, 0 = as fast as possible
    bpf_filter: str | None = None      # optional libpcap filter, e.g. "ip"

    # --- Flow timeouts (seconds of inactivity before a flow is finalized) ---
    tcp_flow_timeout: float = 120.0
    udp_flow_timeout: float = 60.0
    general_flow_timeout: float = 30.0

    # --- Memory bounds ---
    max_active_flows: int = 50_000
    max_history_per_flow: int = 512     # capped packet_sizes/timestamps per flow
    source_history_window_sec: float = 60.0   # window for dest_fanout/port_fanout/flow_count
    max_source_history_entries: int = 10_000  # per-source-IP bound, oldest evicted first

    # --- Logging ---
    log_level: str = "INFO"

    def timeout_for(self, protocol: str) -> float:
        protocol = (protocol or "").upper()
        if protocol == "TCP":
            return self.tcp_flow_timeout
        if protocol == "UDP":
            return self.udp_flow_timeout
        return self.general_flow_timeout


DEFAULT_CONFIG = IngestConfig()
