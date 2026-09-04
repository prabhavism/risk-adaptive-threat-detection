"""
LRU-bounded per-host sequence history, used by predict_interface.py to
feed Heavy DL real recent context per src_ip. Split into its own module
(no tensorflow/xgboost import) so it's unit-testable without the heavy
ML dependencies installed -- see tests/test_person2_fixes.py.

State retention policy (Part 9 of the ingestion-integration brief):
each host's own buffer is bounded to `seq_len` entries (old flows fall
off automatically, oldest first). On top of that, the total *number of
hosts* tracked is bounded to `max_hosts`: once the cap is hit, the
least-recently-touched host is evicted before a new one is admitted.
This only affects Heavy DL's temporal context for a host that hasn't
been seen in a very long time (it starts fresh, same as a brand-new
host) -- it never affects ml_verdict/dl_verdict for the flow currently
being scored.
"""
from collections import OrderedDict, deque

import numpy as np

DEFAULT_MAX_HOSTS = 100_000


class _HostHistory:
    def __init__(self, max_hosts: int = DEFAULT_MAX_HOSTS, seq_len: int = 10):
        self.max_hosts = max_hosts
        self.seq_len = seq_len
        self._data: "OrderedDict[str, deque]" = OrderedDict()

    def append(self, host: str, row: np.ndarray) -> None:
        if host in self._data:
            self._data.move_to_end(host)
        else:
            if len(self._data) >= self.max_hosts:
                self._data.popitem(last=False)  # evict least-recently-used
            self._data[host] = deque(maxlen=self.seq_len)
        self._data[host].append(row)

    def window(self, host: str) -> list:
        if host not in self._data:
            return []
        self._data.move_to_end(host)
        return list(self._data[host])

    def clear(self):
        self._data.clear()

    def __len__(self):
        return len(self._data)
