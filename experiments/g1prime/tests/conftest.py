"""Pytest configuration: make g1prime, e1, and g0 modules importable.

Test files can then do plain imports like::

    import cost_model
    import capacity
    import simulate_concurrency
    import verdict
    import compare_oracle      # from experiments/e1/

The g1prime directory MUST end up at ``sys.path[0]`` because both
``experiments/g0/`` and ``experiments/g1prime/`` contain a file named
``verdict.py``. Without this priority, ``g0/verdict.py`` (G0 verdict
generator) would shadow ``g1prime/verdict.py`` (G1′ verdict generator),
causing ``AttributeError: module 'verdict' has no attribute
'compute_headroom_table'``.

Implementation note: pytest's default ``prepend`` import mode already
inserts the rootdir (``experiments/g1prime``) and the test-file directory
(``experiments/g1prime/tests``) onto ``sys.path`` *before* this conftest
runs. We therefore cannot rely on ``if _p not in sys.path`` (it would
skip re-insertion and leave g1prime at a lower-priority position). We
must explicitly ``remove`` and re-``insert`` to force g1prime to
position 0. We also purge any cached ``verdict`` module from
``sys.modules`` so the next ``import verdict`` re-resolves against the
corrected sys.path.
"""

import importlib
import sys
from pathlib import Path

G1PRIME_DIR: Path = Path(__file__).resolve().parent.parent          # experiments/g1prime/
E1_DIR:      Path = G1PRIME_DIR.parent / "e1"                       # experiments/e1/
G0_DIR:      Path = G1PRIME_DIR.parent / "g0"                       # experiments/g0/

# Insert e1 + g0 (only if not already present).
for _p in (str(G0_DIR), str(E1_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Force g1prime to sys.path[0] regardless of prior insertion by pytest.
_g1prime_str = str(G1PRIME_DIR)
while _g1prime_str in sys.path:
    sys.path.remove(_g1prime_str)
sys.path.insert(0, _g1prime_str)

# If `verdict` was already imported (e.g. g0/verdict.py was picked up by
# an earlier sys.path state), purge it so the next `import verdict` in
# the test files resolves to g1prime/verdict.py at sys.path[0].
if "verdict" in sys.modules:
    _cached = sys.modules.pop("verdict")
    _cached_file = getattr(_cached, "__file__", "<unknown>")
    # Re-import now that sys.path[0] is g1prime.
    importlib.import_module("verdict")
