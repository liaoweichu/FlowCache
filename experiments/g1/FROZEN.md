# G1 Frozen: Diagnostic Negative Result (Protocol-Invalid)

**Frozen Date**: 2026-07-26
**Status**: Diagnostic negative result — protocol-invalid / inconclusive
**Reason**: G1 replay protocol does not match FlowCache's research target:
1. Traces tokenized per-message (not full chat template)
2. Block creation treated as block access (no prefix reuse on resume)
3. Capacity in % budget (10% ≈ 41.4 GiB, unrealistic for 24GB GPU)
4. Concurrency, TTFT, bootstrap statistics incorrect

**Successor**: G1′ (../g1prime/) — physical prefix recompile + correct replay

**Do NOT modify** any file in this directory. It is preserved as diagnostic
evidence of the protocol invalidity that motivated G1′.
