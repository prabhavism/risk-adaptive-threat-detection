# Risk-Adaptive Threat Detection Pipeline

Passive, risk-adaptive AI threat detection prototype. See
`docs/interfaces.md` for the schema/interface contracts between the three
work areas.

## Layers

- **Person 1 — Data layer**: `data/`, PCAP capture, flow extraction, feature
  engineering → `data/flow_features.csv`, `data/labels.csv`
- **Person 2 — Intelligence layer**: `ml_dl/` — XGBoost triage, risk-adaptive
  routing, Light/Heavy DL verification, SHAP → `ml_dl/predict_interface.py`
- **Person 3 — Application layer**: `pipeline/`, `dashboard/`, `tests/` —
  orchestration, correlation, alerts, dashboard, benchmarking (to be added)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Working without real data yet

Generate synthetic data matching the agreed schema and build/test against it:

```bash
python scripts/generate_synthetic_data.py --rows 5000
python -m ml_dl.train_xgboost
pytest tests/ -v
```

When Person 1's real `data/flow_features.csv` lands, just drop it in place
of the synthetic one (same filename, same schema) and re-run the same
commands — no code changes needed.

## Git workflow

- `main` — always working
- `person1-data`, `person2-ml`, `person3-app` — one branch per person
- Open a PR into `main` when your piece is stable; at least one other
  person reviews before merge
- **Never commit** `data/*.pcap`, `data/*.csv`, or `models/*.h5|*.pkl` —
  they're gitignored. Share those via a common Drive folder or Git LFS.
- If you need to change anything in `docs/interfaces.md`, message the team
  first — both other people's code depends on those exact names.

## Day-by-day (Person 2)

| Day | Focus |
|---|---|
| 1 | Repo set up, schema agreed with Person 1, synthetic data generator working |
| 2–3 | XGBoost training script, confidence extraction, threshold routing |
| 4–5 | Light DL architecture + training loop (on synthetic data) |
| 6–7 | Heavy DL architecture + training loop (on synthetic data) |
| 8 | SHAP wrapper + top-feature extraction |
| 9 | `predict_interface.py` wired end-to-end, unit tests passing |
| 10 | Docs for Person 3, dry-run integration with dummy outputs |
| — | *(real data arrives)* swap in real `flow_features.csv`, retrain XGBoost, retune theta, retrain Light/Heavy DL, export final `models/*` |
