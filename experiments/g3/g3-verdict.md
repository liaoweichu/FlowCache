# G3 Verdict Report: Lossless Residency

**Verdict**: ❌ NO-GO

**Main cell**: 1 GiB, concurrency=4

## Conditions

### 1. Overhead Feasibility: ✅ PASS
- Overhead (migrate + restore): 0.00 ms
- Saved prefill: 6009884.08 ms

### 2. p95 TTFT Improvement ≥ 15%: ❌ FAIL
- ❌ vs gdsf: -31.52% (CI=[-29.11%, -8.43%])
- ❌ vs sizecost: -40.73% (CI=[-31.01%, -12.72%])

### 3. Throughput Non-inferior (drop ≤ 5%): ✅ PASS
- ✅ vs gdsf: -0.09%
- ✅ vs sizecost: -0.07%

### 4. Better Than Heuristic (CI > 0): ❌ FAIL
- ❌ vs gdsf: mean=-18.27% (CI=[-29.11%, -8.43%])
- ❌ vs sizecost: mean=-21.76% (CI=[-31.01%, -12.72%])

## All Cells Summary

| Capacity (GiB) | Concurrency | FlowCache p95 TTFT | Best Simple | Best Simple p95 | Improvement | Throughput Δ |
|---:|---:|---:|---|---:|---:|---:|
| 1.0 | 1 | 8116.4 | sizecost | 10240.6 | 20.74% | -0.17% |
| 1.0 | 4 | 107549.6 | sizecost | 76423.6 | -40.73% | -0.07% |
| 1.0 | 8 | 117736.6 | gdsf | 109556.0 | -7.47% | -0.02% |
| 2.0 | 1 | 8118.1 | sizecost | 8244.2 | 1.53% | -0.21% |
| 2.0 | 4 | 8609.4 | gdsf | 22612.1 | 61.93% | -0.20% |
| 2.0 | 8 | 83077.6 | gdsf | 71524.4 | -16.15% | -0.09% |
| 4.0 | 1 | 8118.1 | sizecost | 8119.2 | 0.01% | -0.14% |
| 4.0 | 4 | 8118.1 | sizecost | 8684.0 | 6.52% | -0.14% |
| 4.0 | 8 | 8303.9 | gdsf | 18644.5 | 55.46% | -0.16% |

## Failure Action

按 IDEA §7 G3：路线 A No-Go，转路线 B。
实现保留为工程基线，但不以无损 residency 单独投稿该主张。