"""
Section 37: fails loudly if the ingestion layer's output ever drifts
from what ml_dl.predict_interface.predict() actually expects --
missing feature, extra unexpected feature, wrong type, NaN, or
infinite value. This is the test that protects the Person1<->Person2
schema contract when it's the *ingestion* layer producing the row
instead of Person 1's CSV.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml_dl.config import ORIGINAL_FEATURE_COLUMNS
from ingest.config import IngestConfig
from ingest.flow_builder import FlowBuilder
from ingest.feature_extractor import FeatureExtractor, ALL_OUTPUT_COLUMNS, RAW_CATEGORICAL_COLUMNS
from ingest.schemas import Packet


def _sample_flow_row():
    b = FlowBuilder(IngestConfig())
    packets = [
        Packet(timestamp=0.0, src_ip="10.0.0.5", dst_ip="8.8.8.8", src_port=51000,
               dst_port=53, protocol="UDP", packet_length=80,
               dns_query_name="abc123xyz.example.com", dns_query_length=21,
               dns_record_type="A"),
        Packet(timestamp=0.05, src_ip="8.8.8.8", dst_ip="10.0.0.5", src_port=53,
               dst_port=51000, protocol="UDP", packet_length=120),
    ]
    for p in packets:
        b.add_packet(p)
    flow = b.flush()[0]
    return FeatureExtractor(b).extract(flow)


def test_no_missing_or_extra_columns():
    row = _sample_flow_row()
    expected = set(ALL_OUTPUT_COLUMNS)
    assert set(row.keys()) == expected, (
        f"missing={expected - set(row.keys())} extra={set(row.keys()) - expected}"
    )


def test_all_original_feature_columns_present_and_numeric():
    row = _sample_flow_row()
    for col in ORIGINAL_FEATURE_COLUMNS:
        assert col in row, f"missing required numeric feature: {col}"
        val = row[col]
        assert isinstance(val, (int, float)), f"{col} is not numeric: {type(val)}"
        assert not (isinstance(val, float) and math.isnan(val)), f"{col} is NaN"
        assert not (isinstance(val, float) and math.isinf(val)), f"{col} is infinite"


def test_categorical_columns_are_strings():
    row = _sample_flow_row()
    for col in RAW_CATEGORICAL_COLUMNS:
        assert isinstance(row[col], str), f"{col} should be a string, got {type(row[col])}"


def test_zero_duration_does_not_produce_nan_or_inf():
    # A flow finalized after exactly one packet has duration == 0;
    # packet_rate/byte_rate must safely become 0, not NaN/inf (section 14).
    b = FlowBuilder(IngestConfig())
    b.add_packet(Packet(timestamp=1.0, src_ip="10.0.0.1", dst_ip="10.0.0.2",
                         src_port=1, dst_port=2, protocol="TCP", packet_length=64))
    flow = b.flush()[0]
    row = FeatureExtractor(b).extract(flow)
    assert row["duration"] == 0.0
    assert row["packet_rate"] == 0.0
    assert row["byte_rate"] == 0.0
