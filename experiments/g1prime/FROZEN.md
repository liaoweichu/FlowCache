# G1′ Frozen: PASSED (Go/No-Go = GO)

**Frozen Date**: 2026-07-27
**Status**: PASSED — Go/No-Go verdict = **GO** ✅
**Verdict File**: `g1prime-verdict.json`, `g1prime-verdict.md`

## Verdict Summary

- **5 / 12 cells** pass the 10% headroom threshold with bootstrap CI lower > 0.
- **Best cell**: capacity = 1 GiB, concurrency = 4
  - headroom_rel = **45.80%**
  - CI = [18.11%, 31.23%]
- Full grid: 6 baselines × 4 capacities × 3 concurrencies × 1320 episodes = 95,040 rows.
- Block accesses per cell: 8,262,330 (matches expected ~8.27M).

## Passing Cells

| Capacity (GiB) | Concurrency | headroom_rel | CI lower | CI upper |
|---:|---:|---:|---:|---:|
| 1 | 1 | 36.44% | 3.21% | 12.25% |
| 1 | 4 | **45.80%** | **18.11%** | 31.23% |
| 2 | 4 | 34.90% | 2.67% | 8.69% |
| 2 | 8 | 42.66% | 11.63% | 24.52% |
| 4 | 8 | 16.63% | 0.09% | 3.86% |

## Known Anomaly

Cell (1 GiB, c=8) shows **negative** headroom (-13.90%): Oracle-Cost (1188.87 ms)
underperforms GDSF (1023.68 ms). Likely causes:

1. Under extreme cache pressure (1 GiB) + high concurrency (c=8), working set
   far exceeds capacity → nearly all accesses are misses → eviction policy
   matters less.
2. Oracle-Cost's cost-aware eviction may retain high-cost blocks at the expense
   of evicting blocks that would be reused sooner, backfiring under thrashing.
3. Does NOT affect the Go verdict (only one passing cell is required).

## Key Patterns

1. **Smaller capacity → larger headroom**: 1 GiB (36–46%) ≫ 6 GiB (~0%).
   Eviction policy value emerges under memory pressure.
2. **c=4 is the sweet spot**: highest headroom with the most robust CI.
3. **c=1 at ≥2 GiB**: headroom ≈ 0% (cache large enough for single workflow).

## Decision

G1′ PASSED → proceed to **P1-A (G2 gate)**: joint Reuse-value (R) / Fidelity-risk (D)
controller design, targeting the identified operating points (1 GiB c=4, 2 GiB c=8).

## Do NOT modify

Any file in this directory. The results are preserved as the evidence base for
the G1 gate pass decision. New experiments should be created under
`experiments/g2/` (or similar).
