# Ingestion layer

> The ingestion layer is strictly passive and operates on observed
> packet copies. It does not initiate communication with monitored
> hosts or networks.

Converts network packets (from a PCAP file or a live NIC) into the
flow-level feature records the existing AI engine (`ml_dl/`) already
knows how to score, without changing that engine at all:

```
PCAP / live NIC
      ↓
packet_parser.py    -- raw packet -> normalized Packet
      ↓
flow_builder.py      -- packets -> bidirectional FlowState, with
                         timeouts + bounded memory
      ↓
feature_extractor.py -- finalized FlowState -> raw flow dict
                         (same columns as data/flow_features.csv)
      ↓
ml_dl.predict_interface.predict(flow)   <- UNCHANGED, not modified here
      ↓
pipeline.py           -- wires the above together, returns results
```

## Install

```bash
pip install -r requirements.txt   # now includes scapy
```

PCAP replay needs no special privileges. Live capture needs a raw
socket, so on Linux:

```bash
sudo python scripts/capture_live.py --interface eth0
```

If you don't have root on the box, use PCAP mode instead (this is the
default, reproducible way to run/demo the system).

### OS-specific live-capture requirements (documented, not assumed)

- **Linux (primary target)**: needs `libpcap` installed and either
  root or `CAP_NET_RAW`/`CAP_NET_ADMIN` on the Python interpreter
  (`sudo setcap cap_net_raw,cap_net_admin=eip $(which python3)` as an
  alternative to running the whole process as root). Select the
  interface name from `ip link` (e.g. `eth0`, `ens33`).
- **Windows**: scapy needs [Npcap](https://npcap.com/) installed
  (install with "WinPcap API-compatible mode" checked) and the script
  run from an elevated (Administrator) shell. Interface names on
  Windows are GUIDs/friendly names from `scapy.all.show_interfaces()`,
  not `eth0`.
- **macOS**: needs libpcap (present by default) and `sudo`; interface
  names come from `ifconfig` / `networksetup -listallhardwareports`.

This prototype's tested path is Linux (the intended demo environment,
per the brief). Windows/macOS should work via scapy's cross-platform
support but haven't been exercised here -- don't assume they behave
identically without testing on that OS.

### Demo PCAP fixture

`scripts/generate_demo_pcap.py` writes a tiny, deterministic, harmless
PCAP (`data/demo.pcap`) -- one TCP connection (handshake, a small
HTTP-like exchange, close), one DNS query/response, one ICMP echo --
built with only `struct` (no scapy needed to *generate* it, only to
*read* it back). All addresses are RFC 5737/1918 documentation/private
ranges; nothing here is a real capture or attack traffic:

```bash
python scripts/generate_demo_pcap.py --out data/demo.pcap
```

## PCAP mode (reproducible testing / demo)

```bash
python scripts/replay_pcap.py --pcap data/demo.pcap --speed 1
python scripts/replay_pcap.py --pcap data/demo.pcap --speed 10   # faster
python scripts/replay_pcap.py --pcap data/demo.pcap --speed 0    # max speed, for benchmarking
python scripts/replay_pcap.py --pcap data/demo.pcap --out reports/pcap_results.jsonl
```

Writes one JSON object per finalized flow to a JSONL file as results
arrive (default `reports/pcap_results.jsonl`), with `flow_id`,
`timestamp`, source/destination IP+port, `protocol`, `xgboost_class`,
`xgboost_confidence`, `model_used`, `dl_class`, `dl_confidence`,
`shap_evidence`, and `alert` (a full standardized alert object, or
`null` for benign flows).

## Live mode (passive network interface)

```bash
sudo python scripts/capture_live.py --interface eth0 --flow-timeout 30
```

Streams the same `predict(flow)` result per finalized flow to the
console log. For CSV-sourced streaming instead of live/PCAP capture,
see `scripts/replay_stream.py` (reads `flow_features.csv` row by row).

**What "live capture" actually receives.** Running `capture_live.py` on
a machine only sees traffic that machine's own NIC receives. It does
NOT automatically see another host's (e.g. a server's) traffic just
because the detector is running somewhere on the network. A real
deployment needs an explicit passive copy mechanism feeding the
monitoring NIC:

```text
Client -----> Server              (production traffic, unmodified)
                 |
                 +---- mirrored copy ---->  Monitoring NIC
                                                  |
                                                  v
                                          capture_live.py (this repo)
```

That copy comes from one of: a switch SPAN/mirror port, a physical
network TAP, a virtual switch's port-mirroring feature (e.g. on a
hypervisor), or a hardware data-diode. This prototype's job starts
*after* that copy reaches its NIC -- it does not provide, configure,
or simulate the mirroring itself, and it never establishes a return
path to the monitored client/server (see the passive-only audit below).

## Benchmark

```bash
python scripts/benchmark_ingestion.py --pcap data/demo.pcap
```

## One-command demo

```bash
python demo.py --pcap data/demo.pcap --speed 1
python demo.py --interface eth0
```

## Design decisions worth knowing

- **Flow finalization**: a flow is scored once — on TCP FIN/RST, on
  inactivity timeout, or at PCAP end/flush — not on every packet. This
  is the documented, stability-preferred policy (see `pipeline.py`).
- **Bidirectional flows**: request and response packets are merged into
  one `FlowState` via a direction-independent key (`flow_builder.py:
  normalize_flow_key`), so `byte_ratio`/`packet_ratio` reflect real
  asymmetry instead of two unrelated half-flows.
- **Memory bounds**: active flows are capped (`max_active_flows`),
  per-flow packet history is capped (`max_history_per_flow`), and
  per-source behavioral history is capped + TTL-pruned
  (`max_source_history_entries` / `source_history_window_sec`). None of
  these dictionaries grow without bound. See `ingest/config.py`.
- **No future information**: every feature is computed only from
  packets already seen for that flow, plus that source's own past
  history — see `feature_extractor.py` module docstring.
- **TLS/QUIC**: SNI is read from the plaintext ClientHello (sent
  unencrypted in both TLS 1.2 and TLS 1.3 without ECH) — this is
  reading a plaintext header, not decryption. No TLS/QUIC content is
  ever decrypted. Real JA3/JA3S/JA4 hashing is not implemented in this
  prototype; only handshake *presence* is captured, which is what
  `ml_dl.data_utils.add_engineered_features`'s `has_tls` signal uses.
- **DNS**: only the query name/length/record type are read (standard,
  unencrypted DNS metadata). Encrypted DNS (DoH/DoT) is not decrypted;
  those flows simply won't have DNS fields populated.
- **Schema reuse**: the numeric feature list is imported from
  `ml_dl.config.ORIGINAL_FEATURE_COLUMNS`, not redefined here, so the
  two layers can't silently drift apart.

## Passive-only audit

`ingest/` contains no `send()`, `sendp()`, `sr()`, `sr1()`,
`socket.connect()`, or any other packet-injection / active-scanning
call. `scripts/benchmark_ingestion.py` and the tests only *read*
locally stored PCAPs or synthetic in-memory packets — nothing here
talks to a monitored host.
