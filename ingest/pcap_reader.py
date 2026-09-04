"""
Incremental PCAP reader (section 10). Uses scapy's PcapReader, which
reads one packet at a time from disk rather than loading the whole
file into memory (unlike scapy.rdpcap). Requires scapy:

    pip install scapy

Reading a PCAP never requires root/sudo -- only live capture does.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Iterator, Optional

from ingest.packet_parser import parse_packet
from ingest.schemas import Packet

logger = logging.getLogger("ingest.pcap_reader")


def read_pcap(path: str, speed: float = 1.0) -> Iterator[Packet]:
    """
    Yields normalized Packet objects from a PCAP file, one at a time.

    speed: replay pacing relative to the packets' original capture
    timestamps.
        speed = 1.0  -> approximately real-time replay (default)
        speed = 10.0 -> 10x faster than real-time
        speed = 0    -> no pacing at all, read as fast as possible
                        (used for benchmarking / batch scoring)

    Malformed packets are skipped (logged, not raised) so one bad
    packet can't kill an otherwise-good PCAP (section 32).
    """
    path = str(path)
    if not Path(path).exists():
        raise FileNotFoundError(f"PCAP not found: {path}")

    try:
        from scapy.utils import PcapReader
    except ImportError as e:
        raise ImportError(
            "scapy is required to read PCAPs: pip install scapy"
        ) from e

    prev_ts: Optional[float] = None
    n_read = 0
    n_skipped = 0

    try:
        with PcapReader(path) as reader:
            for raw_pkt in reader:
                try:
                    pkt_time = float(raw_pkt.time)
                except Exception:
                    pkt_time = time.time()

                if speed and speed > 0 and prev_ts is not None:
                    delay = (pkt_time - prev_ts) / speed
                    if delay > 0:
                        time.sleep(delay)
                prev_ts = pkt_time

                parsed = parse_packet(raw_pkt, capture_time=pkt_time)
                n_read += 1
                if parsed is None:
                    n_skipped += 1
                    continue
                yield parsed
    except Exception as e:
        # Truncated/corrupt PCAP tail -- log and stop cleanly rather
        # than propagating a crash (section 32: "truncated PCAP").
        logger.warning("PCAP read ended early (%s): %s", path, e)

    logger.info("PCAP %s: %d packets read, %d skipped (unsupported/malformed)",
                path, n_read, n_skipped)
