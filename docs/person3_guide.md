# Person 3 — Application, Streaming & Dashboard Guide

Your branch: **`person3-app`**  
Your input: `ml_dl/predict_interface.py` (Person 2's output)  
Your deliverable: the **running system** — streaming pipeline, alert engine, dashboard, benchmarks

> **Rule #1** — Your orchestrator calls `predict(flow: dict)` from `ml_dl/predict_interface.py`.
> Never call model internals directly. If Person 2 hasn't finished yet, use the
> **mock stub** in Section 3 below — it matches the exact same interface.

---

## 1. First-Time Setup

```bash
# Clone the repo (if you haven't)
git clone https://github.com/PRABHAVISM/risk-adaptive-threat-detection.git
cd risk-adaptive-threat-detection

# Switch to your branch
git checkout person3-app

# Create a Python virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt
# You will also add to requirements.txt:
# fastapi uvicorn websockets aiofiles tqdm psutil
```

---

## 2. Your Directory — What to Create

```
risk-adaptive-threat-detection/
├── pipeline/                      ← [YOU CREATE] core runtime
│   ├── __init__.py
│   ├── replay.py                  ← PCAP/CSV replay (simulates live feed)
│   ├── orchestrator.py            ← reads flows, calls predict(), emits alerts
│   ├── alert.py                   ← Alert dataclass + serialization
│   └── correlator.py              ← per-host incident correlation
│
├── app/                           ← [YOU CREATE] FastAPI backend
│   ├── __init__.py
│   ├── main.py                    ← FastAPI app, REST + WebSocket endpoints
│   ├── state.py                   ← shared in-memory state (alert buffer, stats)
│   └── schemas.py                 ← Pydantic models for API responses
│
├── dashboard/                     ← [YOU CREATE] frontend (Streamlit or React)
│   └── streamlit_app.py           ← Streamlit dashboard (recommended for speed)
│
├── tests/
│   └── test_pipeline.py           ← already exists, add your tests here
│
├── docker-compose.yml             ← [YOU CREATE] full-stack launch
├── Dockerfile                     ← [YOU CREATE]
└── benchmark.py                   ← [YOU CREATE] throughput + latency test
```

---

## 3. Mock Predict Stub — Start Here Before Person 2 Finishes

Create `pipeline/mock_predict.py` so you can build and test everything independently:

```python
"""
Drop-in stub for ml_dl.predict_interface.predict()
Returns realistic-looking output without any real model.
Replace with: from ml_dl.predict_interface import predict
when Person 2's models are ready.
"""
import random
import time

CLASSES = ["benign","ddos","c2_beaconing","dga_dns_tunnelling",
           "encrypted_malware","recon_scanning","data_exfiltration"]

def predict(flow: dict) -> dict:
    time.sleep(0.002)               # simulate ~2ms inference latency
    verdict = random.choice(CLASSES)
    conf    = random.uniform(0.6, 0.99)
    return {
        "ml_verdict":   verdict,
        "ml_confidence": conf,
        "dl_verdict":   verdict,
        "dl_confidence": min(conf + 0.03, 1.0),
        "model_used":   random.choice(["light", "heavy"]),
        "shap_evidence": [
            {"feature": "packet_rate",  "value": round(random.random(), 3)},
            {"feature": "dest_fanout",  "value": round(random.random(), 3)},
            {"feature": "entropy",      "value": round(random.random(), 3)},
        ]
    }
```

---

## 4. pipeline/alert.py — The Alert Schema

Every detection becomes a standardized `Alert` object. This is what the dashboard displays.

```python
from dataclasses import dataclass, asdict, field
from typing import List, Dict
import uuid, time

SEVERITY_MAP = {
    "benign":             "info",
    "ddos":               "critical",
    "c2_beaconing":       "high",
    "dga_dns_tunnelling": "high",
    "encrypted_malware":  "medium",
    "recon_scanning":     "medium",
    "data_exfiltration":  "critical",
}

@dataclass
class Alert:
    alert_id:       str   = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp:      float = field(default_factory=time.time)
    flow_id:        str   = ""
    src_ip:         str   = ""
    dst_ip:         str   = ""
    src_port:       int   = 0
    dst_port:       int   = 0
    protocol:       str   = ""
    threat_class:   str   = "benign"
    confidence:     float = 0.0
    risk_score:     float = 0.0     # confidence * severity_weight
    severity:       str   = "info"
    model_used:     str   = "light"
    evidence:       List[Dict] = field(default_factory=list)  # shap_evidence

    def to_dict(self) -> dict:
        return asdict(self)

SEVERITY_WEIGHT = {"info": 0.1, "medium": 0.5, "high": 0.8, "critical": 1.0}

def flow_to_alert(flow: dict, prediction: dict) -> Alert:
    threat  = prediction["ml_verdict"]
    conf    = prediction["ml_confidence"]
    sev     = SEVERITY_MAP.get(threat, "info")
    return Alert(
        flow_id      = flow.get("flow_id", ""),
        src_ip       = flow.get("src_ip", ""),
        dst_ip       = flow.get("dst_ip", ""),
        src_port     = flow.get("src_port", 0),
        dst_port     = flow.get("dst_port", 0),
        protocol     = flow.get("protocol", ""),
        threat_class = threat,
        confidence   = conf,
        risk_score   = round(conf * SEVERITY_WEIGHT[sev], 3),
        severity     = sev,
        model_used   = prediction["model_used"],
        evidence     = prediction["shap_evidence"],
    )
```

---

## 5. pipeline/correlator.py — Host-Level Incident Correlation

Multiple alerts about the same host should be grouped into an **Incident**.

```python
"""
Correlates individual alerts by src_ip into higher-level incidents.
Example: recon → suspicious DNS → beaconing → exfil = one Incident for that host.
"""
from collections import defaultdict
from typing import List
import time

INCIDENT_WINDOW = 300   # seconds — group alerts within 5 minutes into one incident

class Correlator:
    def __init__(self):
        # host_ip → list of Alert objects
        self._host_alerts: dict = defaultdict(list)

    def ingest(self, alert) -> dict | None:
        """Add alert. Returns an Incident dict if correlation threshold is met."""
        if alert.threat_class == "benign":
            return None

        self._host_alerts[alert.src_ip].append(alert)
        self._purge_old(alert.src_ip, INCIDENT_WINDOW)

        host_alerts = self._host_alerts[alert.src_ip]
        classes_seen = {a.threat_class for a in host_alerts}
        max_risk     = max(a.risk_score for a in host_alerts)

        # Escalate to incident if ≥2 distinct threat types from same host
        if len(classes_seen) >= 2 or max_risk > 0.85:
            return {
                "incident_id":   alert.src_ip,
                "src_ip":        alert.src_ip,
                "alert_count":   len(host_alerts),
                "threat_classes": list(classes_seen),
                "max_risk_score": max_risk,
                "alerts":        [a.to_dict() for a in host_alerts],
            }
        return None

    def _purge_old(self, host: str, window: float):
        cutoff = time.time() - window
        self._host_alerts[host] = [
            a for a in self._host_alerts[host] if a.timestamp >= cutoff
        ]
```

---

## 6. pipeline/orchestrator.py — The Main Loop

```python
"""
Reads flows one by one (from CSV replay or live feed),
calls predict(), creates Alerts, correlates into Incidents,
and pushes to the FastAPI state for the dashboard.
"""
import asyncio
import pandas as pd
from pipeline.alert import flow_to_alert
from pipeline.correlator import Correlator

# Swap this import when Person 2's model is ready:
from pipeline.mock_predict import predict
# from ml_dl.predict_interface import predict   ← real model

async def run(csv_path: str, state):
    correlator = Correlator()
    df = pd.read_csv(csv_path)

    for _, row in df.iterrows():
        flow = row.to_dict()
        prediction = predict(flow)
        alert      = flow_to_alert(flow, prediction)

        state.add_alert(alert)
        incident = correlator.ingest(alert)
        if incident:
            state.add_incident(incident)

        await asyncio.sleep(0)    # yield to event loop (non-blocking)
```

---

## 7. pipeline/replay.py — PCAP / CSV Replay

```python
"""
Replays a flow_features.csv at a controlled rate to simulate live traffic.
Replace with a live tshark/Zeek feed later — orchestrator.py stays the same.
"""
import asyncio, time
import pandas as pd

async def replay_csv(csv_path: str, flows_per_sec: int = 1000):
    df = pd.read_csv(csv_path)
    interval = 1.0 / flows_per_sec
    for _, row in df.iterrows():
        yield row.to_dict()
        await asyncio.sleep(interval)
```

---

## 8. app/main.py — FastAPI Backend

```python
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from app.state import AppState
from pipeline.orchestrator import run
import asyncio

app = FastAPI(title="Threat Detection API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
state = AppState()

@app.on_event("startup")
async def startup():
    asyncio.create_task(run("data/flow_features.csv", state))

@app.get("/alerts")
def get_alerts(limit: int = 50):
    return state.get_recent_alerts(limit)

@app.get("/incidents")
def get_incidents():
    return state.get_incidents()

@app.get("/stats")
def get_stats():
    return state.get_stats()

@app.websocket("/ws")
async def websocket_feed(ws: WebSocket):
    await ws.accept()
    last = 0
    while True:
        alerts = state.get_recent_alerts(100)
        if len(alerts) > last:
            for a in alerts[last:]:
                await ws.send_json(a)
            last = len(alerts)
        await asyncio.sleep(0.5)
```

Run the backend:
```bash
uvicorn app.main:app --reload --port 8000
```

---

## 9. dashboard/streamlit_app.py — Dashboard

```python
import streamlit as st
import requests, time, pandas as pd

st.set_page_config(page_title="Threat Detection Dashboard", layout="wide")
st.title("🛡️ Risk-Adaptive Threat Detection")

API = "http://localhost:8000"

placeholder = st.empty()
while True:
    stats   = requests.get(f"{API}/stats").json()
    alerts  = requests.get(f"{API}/alerts?limit=100").json()

    with placeholder.container():
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Flows/sec",    stats.get("flows_per_sec", 0))
        c2.metric("Active Alerts",stats.get("alert_count", 0))
        c3.metric("Incidents",    stats.get("incident_count", 0))
        c4.metric("Avg Confidence", f"{stats.get('avg_confidence', 0):.0%}")

        st.subheader("Recent Alerts")
        if alerts:
            df = pd.DataFrame(alerts)
            st.dataframe(df[["timestamp","src_ip","dst_ip","threat_class",
                              "confidence","severity","risk_score","model_used"]],
                         use_container_width=True)

            selected = st.selectbox("Select alert to see evidence", df["alert_id"])
            row = df[df["alert_id"] == selected].iloc[0]
            st.json(row["evidence"])

    time.sleep(2)
    placeholder.empty()
```

Run the dashboard:
```bash
streamlit run dashboard/streamlit_app.py
```

---

## 10. benchmark.py — Throughput & Latency

```python
"""
Measures: flows/sec, p50/p95/p99 latency, CPU/RAM, % routed to heavy model.
Run: python benchmark.py --rows 10000
"""
import time, argparse, statistics, psutil
import pandas as pd
from pipeline.mock_predict import predict  # swap for real when ready

def benchmark(csv_path: str, n_rows: int):
    df = pd.read_csv(csv_path).head(n_rows)
    latencies = []
    heavy_count = 0

    proc = psutil.Process()
    cpu_before = proc.cpu_percent()
    mem_before = proc.memory_info().rss / 1024**2

    t0 = time.perf_counter()
    for _, row in df.iterrows():
        t_start = time.perf_counter()
        result  = predict(row.to_dict())
        latencies.append((time.perf_counter() - t_start) * 1000)  # ms
        if result["model_used"] == "heavy":
            heavy_count += 1
    elapsed = time.perf_counter() - t0

    print(f"Flows processed : {n_rows}")
    print(f"Total time      : {elapsed:.2f}s")
    print(f"Throughput      : {n_rows/elapsed:.0f} flows/sec")
    print(f"Latency p50     : {statistics.median(latencies):.2f} ms")
    print(f"Latency p95     : {sorted(latencies)[int(0.95*len(latencies))]:.2f} ms")
    print(f"Latency p99     : {sorted(latencies)[int(0.99*len(latencies))]:.2f} ms")
    print(f"Heavy model %   : {100*heavy_count/n_rows:.1f}%")
    print(f"RAM usage       : {proc.memory_info().rss/1024**2 - mem_before:.1f} MB delta")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",  default="data/flow_features.csv")
    ap.add_argument("--rows", type=int, default=10000)
    args = ap.parse_args()
    benchmark(args.csv, args.rows)
```

---

## 11. docker-compose.yml

```yaml
version: "3.9"
services:
  api:
    build: .
    ports:
      - "8000:8000"
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000
    volumes:
      - ./data:/app/data:ro
      - ./models:/app/models:ro

  dashboard:
    build: .
    ports:
      - "8501:8501"
    command: streamlit run dashboard/streamlit_app.py --server.address 0.0.0.0
    depends_on:
      - api
```

---

## 12. Day-by-Day Plan

| Day | Focus |
|---|---|
| 1 | Clone repo, read `docs/interfaces.md`, run synthetic data generator, inspect schema |
| 2 | Create `pipeline/mock_predict.py` stub; build `pipeline/alert.py`; write alert unit tests |
| 3 | Build `pipeline/correlator.py`; test correlation with multi-alert scenarios |
| 4 | Build `pipeline/orchestrator.py` + `pipeline/replay.py`; run end-to-end with mock |
| 5 | Build `app/main.py` (FastAPI); test REST endpoints with curl / Postman |
| 6 | Build `dashboard/streamlit_app.py`; display live alert feed |
| 7 | Wire WebSocket feed from backend to dashboard |
| 8 | Write `benchmark.py`; measure throughput + latency on synthetic data |
| 9 | Write `docker-compose.yml` + `Dockerfile`; test full stack in Docker |
| 10 | Swap `mock_predict` → real `ml_dl.predict_interface.predict`; rerun benchmarks; open PR |

---

## 13. Git Workflow

```bash
# Start of every session
git checkout person3-app
git pull origin person3-app

# After doing work
git add pipeline/ app/ dashboard/ benchmark.py
git commit -m "feat(app): add host correlation engine with 5-min incident window"
git push origin person3-app

# When a piece is stable → open a Pull Request on GitHub:
#   person3-app → main
# Tag Person 2 to review (your orchestrator depends on their predict interface)
```

> **Shared files** (`requirements.txt`, `docs/interfaces.md`) — message the team before editing.
