# E1 Workload Characterization Report

## 1. Workflow Structure

**Number of workflows**: 5

| Metric | Mean | Median | P95 | P99 | Min | Max |
|--------|------|--------|-----|-----|-----|-----|
| length | 3 | 3 | 5.4 | 5.88 | 0 | 6 |
| depth | 0.2 | 0 | 0.8 | 0.96 | 0 | 1 |
| width | 0.2 | 0 | 0.8 | 0.96 | 0 | 1 |
| branch_rate | 0.03 | 0.0 | 0.13 | 0.16 | 0.0 | 0.1667 |
| tool_wait_duration_ms | 1034.25 | 0.0 | 4137.0 | 4964.4 | 0.0 | 5171.25 |

### Per-Workflow

| workflow_id | length | depth | width | branch_rate | tool_wait_duration_ms |
|-------------|--------|-------|-------|-------------|-----------------------|
| unknown | 0 | 0 | 0 | 0.0 | 0.0 |
| airline-1 | 3 | 0 | 0 | 0.0 | 0.0 |
| airline-2 | 3 | 0 | 0 | 0.0 | 0.0 |
| retail-1 | 6 | 1 | 1 | 0.1667 | 5171.25 |
| retail-2 | 3 | 0 | 0 | 0.0 | 0.0 |

## 2. Exact-Prefix Overlap

- **Overlap Ratio**: 0.2525
- **Total Tokens**: 1117
- **Shared Tokens**: 282
- **Total Unique Blocks**: 76
- **Shared Blocks (>=2 workflows)**: 18
- **Workflow Pairs (for LCP)**: 10

### LCP Token Distribution

| Statistic | Value (tokens) |
|-----------|----------------|
| Mean | 28.8 |
| Median | 0.0 |
| P95 | 144.0 |
| P99 | 144.0 |
| Min | 0 |
| Max | 144 |
| Count | 10 |

## 3. Next-Use Distance Distribution

| Statistic | Value (global steps) |
|-----------|----------------------|
| Mean | 4.5 |
| Median | 4.5 |
| P95 | 6.0 |
| P99 | 6.0 |
| Max | 6 |
| Count (multi-use blocks) | 18 |
| Singleton Blocks | 58 |
| Multi-Use Blocks | 18 |
| Total Unique Blocks | 76 |

## 4. Block Working-Set Size and KV/VRAM Ratio

- **Model**: Qwen2.5-7B-Instruct (28 layers, 28 Q heads, 4 KV heads, head_dim=128, BF16)
- **Per-Block KV**: 0.875 MB (block_size=16 tokens)
- **Working Set Size (peak)**: 39 blocks
- **KV Memory Estimate**: 0.0333 GB
- **KV/VRAM Ratio**: 0.0014 (24.0 GB VRAM)

### Per-Workflow Peak Active Blocks

| workflow_id | peak_active_blocks |
|-------------|--------------------|
| unknown | 0 |
| airline-1 | 14 |
| airline-2 | 17 |
| retail-1 | 39 |
| retail-2 | 24 |
