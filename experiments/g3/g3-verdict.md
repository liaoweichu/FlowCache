# G3 Verdict Report: Lossless Residency

**Verdict**: ❌ NO-GO

**Main cell**: 1 GiB, concurrency=4

## Conditions

### 1. Overhead Feasibility: ✅ PASS
- Overhead (migrate + restore): 0.00 ms
- Saved prefill: 10486742.81 ms

### 2. p95 TTFT Improvement ≥ 15%: ❌ FAIL
- ❌ vs gdsf: -40.04% (CI=[-40.04%, -40.04%])
- ❌ vs sizecost: -69.11% (CI=[-69.11%, -69.11%])

### 3. Throughput Non-inferior (drop ≤ 5%): ✅ PASS
- ✅ vs gdsf: +0.83%
- ✅ vs sizecost: +0.83%

### 4. Better Than Heuristic (CI > 0): ❌ FAIL
- ❌ vs gdsf: mean=-40.04% (CI=[-40.04%, -40.04%])
- ❌ vs sizecost: mean=-69.11% (CI=[-69.11%, -69.11%])

## All Cells Summary

| Capacity (GiB) | Concurrency | FlowCache p95 TTFT | Best Simple | Best Simple p95 | Improvement | Throughput Δ |
|---:|---:|---:|---|---:|---:|---:|
| 1.0 | 1 | 9141.7 | sizecost | 13677.7 | 33.16% | +0.00% |
| 1.0 | 4 | 168558.1 | sizecost | 99676.3 | -69.11% | +0.83% |
| 1.0 | 8 | 124638.3 | gdsf | 117283.6 | -6.27% | +0.00% |
| 2.0 | 1 | 9141.7 | sizecost | 9141.7 | 0.00% | +0.00% |
| 2.0 | 4 | 22003.6 | gdsf | 33979.2 | 35.24% | +0.00% |
| 2.0 | 8 | 103520.4 | sizecost | 71873.4 | -44.03% | +0.00% |
| 4.0 | 1 | 9141.7 | sizecost | 9141.7 | 0.00% | +0.00% |
| 4.0 | 4 | 9745.7 | gdsf | 10759.8 | 9.43% | +0.00% |
| 4.0 | 8 | 22993.1 | sizecost | 15468.8 | -48.64% | +0.00% |

## Failure Action

按 IDEA §7 G3：路线 A No-Go，转路线 B。
实现保留为工程基线，但不以无损 residency 单独投稿该主张。