# Person 1 — Data Creation & Feature Engineering Guide

Your branch: **`person1-data`**  
Your output that Person 2 and 3 depend on: **`data/flow_features.csv`**

> **Rule #1** — Do NOT change any column name in `data/flow_features.csv` without
> messaging the whole team first. Person 2's entire ML pipeline is built on those
> exact names. See `docs/interfaces.md` for the locked schema.

---

## 1. First-Time Setup

```bash
# Clone the repo (if you haven't)
git clone https://github.com/PRABHAVISM/risk-adaptive-threat-detection.git
cd risk-adaptive-threat-detection

# Switch to your branch
git checkout person1-data

# Create a Python virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

---

## 2. Your Directory — What to Create

```
risk-adaptive-threat-detection/
├── data/
│   ├── .gitkeep              ← already there (keeps folder in git)
│   ├── flow_features.csv     ← YOUR MAIN OUTPUT (gitignored, share via Drive)
│   ├── labels.csv            ← optional separate label file (also gitignored)
│   └── raw/                  ← local only, never committed
│       └── *.pcap
│
└── scripts/                  ← YOUR CODE GOES HERE
    ├── generate_synthetic_data.py   ← already exists (Person 2 wrote this)
    ├── capture_metadata.py          ← [YOU CREATE] log experiment metadata
    ├── extract_flows.py             ← [YOU CREATE] PCAP → flow records
    ├── feature_engineer.py          ← [YOU CREATE] flows → feature vectors
    ├── label_flows.py               ← [YOU CREATE] attach ground-truth labels
    └── split_dataset.py             ← [YOU CREATE] train/val/test split
```

---

## 3. Traffic Generation Checklist

### Benign Traffic
| Tool | Command example | Notes |
|---|---|---|
| iperf3 | `iperf3 -c <dst> -t 60 -b 100M` | TCP/UDP bulk transfer |
| curl/wget | normal HTTP/HTTPS browsing loops | Web traffic |
| Ostinato / TRex | GUI or script | Realistic mixed traffic |

### Attack Traffic (controlled lab only)
| Attack class | Label string | Tool/method |
|---|---|---|
| SYN flood / UDP flood | `ddos` | `hping3 -S --flood -p 80 <target>` |
| Port/host scanning | `recon_scanning` | `nmap -sS -p 1-1024 <range>` |
| DNS tunnelling | `dga_dns_tunnelling` | `iodine`, `dnscat2` |
| DGA simulation | `dga_dns_tunnelling` | script that queries randomly generated domains |
| C2 beaconing | `c2_beaconing` | script with periodic callbacks (sleep + HTTP/HTTPS) |
| Unusual encrypted traffic | `encrypted_malware` | custom TLS client with unusual cipher / JA3 |
| Data exfiltration | `data_exfiltration` | large outbound transfer after benign inbound |

> **Capture metadata** for every experiment — you'll need it to label flows:
> ```
> experiment_id, attack_type, src_ip, dst_ip, start_time, end_time, traffic_rate_mbps
> ```
> Save this as `data/raw/experiment_log.csv` (gitignored, share via Drive).

---

## 4. scripts/extract_flows.py — What to Build

**Input**: a PCAP file path  
**Output**: a pandas DataFrame / CSV with one row per flow (5-tuple based)

```python
# Skeleton — implement this
def extract_flows(pcap_path: str) -> pd.DataFrame:
    """
    Use Zeek logs OR tshark to extract flow records.

    Option A — Zeek (recommended):
        Run zeek on the PCAP, parse conn.log
        zeek -r capture.pcap

    Option B — tshark:
        tshark -r capture.pcap -T fields \
            -e ip.src -e ip.dst -e tcp.srcport -e tcp.dstport \
            -e ip.proto -e frame.time_epoch -e frame.len \
            -E header=y -E separator=, > flows_raw.csv

    Return columns (minimum):
        src_ip, dst_ip, src_port, dst_port, protocol,
        start_time, end_time, packet_count, byte_count,
        tcp_flags (if TCP), dns_query (if DNS), tls_info (if TLS)
    """
```

---

## 5. scripts/feature_engineer.py — Every Feature Explained

This is the most important script. For each flow row, compute:

### Basic Flow Features
| Column | How to calculate |
|---|---|
| `flow_id` | `uuid.uuid4()` or hash of 5-tuple + timestamp |
| `duration` | `end_time - start_time` (seconds) |
| `packet_count` | count of packets in flow |
| `byte_count` | sum of packet lengths |
| `packet_rate` | `packet_count / duration` |
| `byte_rate` | `byte_count / duration` |
| `avg_packet_size` | `byte_count / packet_count` |
| `byte_ratio` | `outbound_bytes / (inbound_bytes + 1)` — asymmetry signal |
| `packet_ratio` | `outbound_pkts / (inbound_pkts + 1)` |

### Windowed / Behavioural Features
> These require a **sliding time-window aggregation** over all flows from the same `src_ip`.
> Use 30-second or 1-minute windows.

| Column | How to calculate | Attack signal |
|---|---|---|
| `dest_fanout` | distinct `dst_ip` values seen from this `src_ip` in window | recon scanning |
| `port_fanout` | distinct `dst_port` values seen from this `src_ip` in window | port scanning |
| `flow_count` | number of flows from this `src_ip` in window | flood / beaconing |
| `entropy` | Shannon entropy of payload bytes (or byte-value histogram) | DGA / tunnelling |

### Inter-Arrival Time Features
```python
# iat = list of time gaps between consecutive packets in this flow
import numpy as np
iat_mean = np.mean(iat)
iat_std  = np.std(iat)

# Coefficient of Variation — key beaconing signal
# Low CV = very regular timing = likely beacon
beacon_cv = iat_std / (iat_mean + 1e-9)

# periodicity_score: map CV to 0-1 where 1 = highly periodic
periodicity_score = float(np.exp(-beacon_cv))   # or use FFT-based method
```

### DNS Features (set to 0 / "" if not a DNS flow)
```python
import math

def char_entropy(s: str) -> float:
    freq = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    return -sum((v/len(s)) * math.log2(v/len(s)) for v in freq.values())

dns_query_length = len(query_name)           # number of characters
dns_entropy      = char_entropy(query_name)  # high entropy → DGA/tunnel
dns_record_type  = "A" | "AAAA" | "TXT" | "NS" | ""
```

### TLS / JA3 Features (set to "" if not a TLS flow)
```python
# Use dpkt, pyshark, or Zeek's ssl.log
tls_ja3   = "<md5 of client hello params>"   # use ja3 library
tls_ja3s  = "<md5 of server hello params>"
tls_ja4   = "<ja4 fingerprint if available>"
tls_sni   = "<server name indication>"
```

> **Python libraries to use**: `dpkt`, `pyshark`, `scapy`, `ja3` (pip install ja3)  
> **Zeek scripts**: `scripts/zeek/` (create this folder and add any custom scripts)

---

## 6. scripts/label_flows.py — Ground-Truth Labelling

```python
# Logic:
# 1. Load experiment_log.csv (src_ip, dst_ip, start_time, end_time, attack_type)
# 2. For each flow, check if flow.start_time falls within an experiment window
#    AND flow.src_ip / dst_ip matches the experiment's src/dst
# 3. If yes → label = experiment attack_type (mapped to one of the 7 class strings)
# 4. If no  → label = "benign"

LABEL_MAP = {
    "syn_flood":        "ddos",
    "udp_flood":        "ddos",
    "port_scan":        "recon_scanning",
    "host_scan":        "recon_scanning",
    "dns_tunnel":       "dga_dns_tunnelling",
    "dga":              "dga_dns_tunnelling",
    "c2_beacon":        "c2_beaconing",
    "tls_malware":      "encrypted_malware",
    "exfil":            "data_exfiltration",
    "benign":           "benign",
}
```

**Important**: Label at the **flow level**, not at the PCAP level.
A single PCAP can contain both benign and attack traffic — that's intentional.

---

## 7. scripts/split_dataset.py — Train / Val / Test Split

```python
# Split by SCENARIO not by random row — otherwise train and test look identical
# e.g.
#   Scenario A captures (morning) → train
#   Scenario B captures (afternoon) → validation
#   Scenario C captures (different day / different attack params) → test

# This prevents data leakage from nearly identical traffic windows
from sklearn.model_selection import GroupShuffleSplit

# group by experiment_id so flows from the same experiment stay together
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
```

Output files (all gitignored, share via Drive):
- `data/train.csv`
- `data/val.csv`
- `data/test.csv`

---

## 8. The Exact Output Schema (do not deviate)

The final `data/flow_features.csv` **must** match `docs/interfaces.md` exactly.
Run this to verify before sharing:

```python
import pandas as pd
from ml_dl.config import FEATURE_COLUMNS

REQUIRED = ["flow_id","src_ip","dst_ip","src_port","dst_port","protocol",
            "duration","packet_count","byte_count","packet_rate","byte_rate",
            "avg_packet_size","byte_ratio","packet_ratio","dest_fanout",
            "port_fanout","flow_count","entropy","dns_query_length",
            "dns_entropy","dns_record_type","tls_ja3","tls_ja3s","tls_ja4",
            "tls_sni","iat_mean","iat_std","periodicity_score","label"]

df = pd.read_csv("data/flow_features.csv")
missing = set(REQUIRED) - set(df.columns)
assert not missing, f"Missing columns: {missing}"
print("Schema OK ✓")
```

---

## 9. Day-by-Day Plan

| Day | Focus |
|---|---|
| 1 | Clone repo, set up environment, read `docs/interfaces.md`, generate synthetic data with `python scripts/generate_synthetic_data.py --rows 100` and inspect the schema |
| 2 | Set up lab: configure iperf3 for benign captures; test hping3 for SYN flood; write `capture_metadata.py` |
| 3 | Capture benign + DDoS PCAPs; write `extract_flows.py` using tshark or Zeek |
| 4 | Capture scan + beaconing PCAPs; extend `extract_flows.py` to IAT and DNS fields |
| 5 | Capture DGA + DNS tunnel + exfil PCAPs; implement `feature_engineer.py` |
| 6 | Implement `label_flows.py`; verify labels match experiment log |
| 7 | Run full pipeline: PCAP → flows → features → labels → validate schema |
| 8 | Write `split_dataset.py`; generate train/val/test splits |
| 9 | Upload `flow_features.csv`, `train.csv`, `val.csv`, `test.csv` to shared Drive; notify Person 2 |
| 10 | Code review of scripts, clean up, push final code to `person1-data`, open PR |

---

## 10. Git Workflow

```bash
# Start of every session
git checkout person1-data
git pull origin person1-data

# After doing work
git add scripts/extract_flows.py scripts/feature_engineer.py   # only your files
git commit -m "feat(data): implement tshark flow extraction with IAT features"
git push origin person1-data

# When a script is complete and tested → open a Pull Request on GitHub:
#   person1-data → main
# Tag Person 2 to review (they depend on your output schema)
```

> **Never add**: `data/*.csv`, `data/*.pcap`, `data/raw/` — these are gitignored.
> Share via the team's **Google Drive** folder.
