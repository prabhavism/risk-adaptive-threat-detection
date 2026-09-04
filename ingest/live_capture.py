"""
Live, strictly passive packet capture from a network interface
(section 11/31). Requires scapy and (on Linux) root/CAP_NET_RAW to
open a raw socket:

    sudo python scripts/capture_live.py --interface eth0

PASSIVE-ONLY GUARANTEE: this module only ever calls scapy's `sniff()`,
which opens the interface in receive-only mode. There is no send(),
sendp(), sr(), sr1(), or socket.connect() anywhere in ingest/. This
process never originates traffic toward the monitored network -- see
ingest/README.md for the full audit statement.

Linux is the primary target for the demo (Ubuntu monitoring VM per the
brief); scapy also runs on macOS/Windows with the right driver
(Npcap on Windows) but that isn't the tested path here.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

from ingest.packet_parser import parse_packet
from ingest.schemas import Packet

logger = logging.getLogger("ingest.live_capture")


def capture_live(interface: str, on_packet: Callable[[Packet], None],
                  bpf_filter: Optional[str] = None, packet_count: int = 0):
    """
    Sniffs `interface` and calls on_packet(parsed_packet) for every
    supported packet, forever (or until `packet_count` packets, 0 =
    unbounded). Blocking call -- run it in a loop / dedicated thread
    from the caller (see scripts/capture_live.py).
    """
    try:
        from scapy.all import sniff
    except ImportError as e:
        raise ImportError(
            "scapy is required for live capture: pip install scapy"
        ) from e

    def _on_raw_packet(raw_pkt):
        parsed = parse_packet(raw_pkt)
        if parsed is not None:
            on_packet(parsed)

    logger.info(
        "Starting PASSIVE live capture on interface=%s filter=%r "
        "(receive-only; this process never sends packets)",
        interface, bpf_filter,
    )
    sniff(
        iface=interface,
        prn=_on_raw_packet,
        filter=bpf_filter,
        store=False,       # never buffer captured packets in memory
        count=packet_count or 0,
    )
