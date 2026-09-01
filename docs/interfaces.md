# Interface Contracts

These are the two contracts that let all three of us work in parallel without
waiting on each other. If either contract changes, ping the whole team before
changing code — everything downstream depends on the exact column names /
field names below.

---

## 1. Person 1 → Person 2: `data/flow_features.csv`

One row per network flow. Person 2's synthetic data generator
(`scripts/generate_synthetic_data.py`) produces a file with exactly this
schema so Person 2 can build against it before real data exists.

| Column               | Type    | Notes                                            |
|----------------------|---------|---------------------------------------------------|
| flow_id              | string  | unique id                                          |
| src_ip               | string  | 5-tuple                                            |
| dst_ip               | string  | 5-tuple                                            |
| src_port              | int     | 5-tuple                                            |
| dst_port              | int     | 5-tuple                                            |
| protocol              | string  | tcp / udp / icmp                                   |
| duration              | float   | seconds                                            |
| packet_count          | int     |                                                     |
| byte_count            | int     |                                                     |
| packet_rate           | float   | packets/sec                                        |
| byte_rate             | float   | bytes/sec                                          |
| avg_packet_size       | float   |                                                     |
| byte_ratio            | float   | src↔dst byte asymmetry                             |
| packet_ratio          | float   | src↔dst packet asymmetry                           |
| dest_fanout           | int     | distinct destinations from this src in window      |
| port_fanout           | int     | distinct dest ports from this src in window        |
| flow_count            | int     | related flows in window (for scan/beacon detection)|
| entropy               | float   | payload/byte entropy                               |
| dns_query_length      | float   | 0 if not DNS                                       |
| dns_entropy           | float   | 0 if not DNS                                       |
| dns_record_type       | string  | empty if not DNS                                   |
| tls_ja3               | string  | empty if not TLS                                   |
| tls_ja3s              | string  | empty if not TLS                                   |
| tls_ja4               | string  | empty if not TLS                                   |
| tls_sni                | string  | empty if not TLS                                   |
| iat_mean               | float   | inter-arrival time mean                            |
| iat_std                | float   | inter-arrival time std dev                         |
| periodicity_score      | float   | 0–1, higher = more periodic (beaconing signal)      |
| label                  | string  | one of the 7 classes below                          |

**`label` values (exactly these strings):**
`benign`, `ddos`, `c2_beaconing`, `dga_dns_tunnelling`, `encrypted_malware`,
`recon_scanning`, `data_exfiltration`

`data/labels.csv` should map `flow_id → label` if labels are kept separate
from features; either is fine as long as `flow_id` is the join key.

---

## 2. Person 2 → Person 3: prediction interface

`ml_dl/predict_interface.py` exposes one function:

```python
def predict(flow: dict) -> dict:
    """
    flow: a single row from flow_features.csv as a dict.
    Returns:
    {
        "ml_verdict": "ddos",              # str, one of the 7 classes
        "ml_confidence": 0.93,             # float 0-1
        "dl_verdict": "ddos",              # str, one of the 7 classes
        "dl_confidence": 0.97,             # float 0-1
        "model_used": "light",             # "light" or "heavy"
        "shap_evidence": [                 # top contributing features
            {"feature": "packet_rate", "value": 0.41},
            {"feature": "dest_fanout", "value": 0.22}
        ]
    }
    """
```

This is the exact dict Person 3's orchestrator, correlation engine, and
alert schema will consume. Field names above are final — if Person 2 needs
to rename anything, flag it to Person 3 first.
