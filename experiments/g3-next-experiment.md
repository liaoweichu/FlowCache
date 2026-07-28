# G3-P1：因果 Cost-Aware 准入、选择性迁移与下一实验

> 更新：2026-07-28  
> 实现状态：`CAUSAL-GPU-ADMISSION+SELECTIVE-MIGRATION-IMPLEMENTED`  
> 证据状态：`PROTOCOL-INCOMPLETE`，不是 GO，也不是 NO-GO  
> 下一实验：**task-grouped validation 上冻结 admission 参数，再跑 2 GiB/c=4 全 trace**  
> 禁止动作：用局部 smoke 调参后汇报 test，或直接进入 9-cell/G4

## 1. 对当前结果的解释

当前两行结果不能证明 FlowCache 已优于 oracle：

| 策略 | modeled p95（旧口径） | hit rate | migrate / restore | replay wall |
|---|---:|---:|---:|---:|
| Oracle-Cost（GPU-only） | 8,626.1 ms | 0.936 | 0 / 0 | 35.5 s |
| Always-Migrate two-tier | 8,111.0 ms | 0.941 | 641,863 / 159,210 | 1,283.1 s |

旧口径下 p95 数值改善约 6.0%、hit rate 增加 0.5 个百分点，但比较不公平且计费不完整：

1. FlowCache 额外使用约 `2× GPU capacity` 的 CPU tier，Oracle-Cost 只有 GPU tier。
2. 旧 p95 只累计 miss-prefill，没有逐请求加入 D2H/H2D 与 controller 成本。
3. 旧 `throughput = request 数 / trace 到达窗口` 对所有策略相同，只是 offered load。
4. 1,283.1 s 来自 Python replay 每次迁移线性扫描 GPU/CPU cache；它是实验脚本复杂度，不是在线推理延迟。
5. 旧 controller 对每个 GPU victim 都迁移，属于容量扩展 baseline，不是 selective value-aware policy；该行为现已独立保留为 `flowcache_always_migrate` 消融。

801,073 次移动对应约 684.5 GiB 建模流量；按 exact block-size 实测中位数估算，串行 DMA 总计约 43.2 s，远小于 1,283.1 s。这进一步说明主要瓶颈是模拟器扫描。

## 2. 已完成的修复

- GPU LRU victim：全扫描改为 `OrderedDict` 首元素，O(1)。
- CPU lowest-R victim：改为带版本号的惰性 heap，摊还 O(log N)。
- D2H/H2D：exact-size 实测中位数优先；分段插值；禁止负成本。
- 逐请求计费：miss-prefill + migrate + restore + modeled controller cost。
- 指标命名：open-loop 输出 `modeled_cache_delay` 和 `offered_load`；真实 TTFT/throughput 标记为无效。
- verdict：协议缺项时输出 `PROTOCOL-INCOMPLETE`，不再误触发路线切换。
- 聚合：hits、misses、saved-prefill、miss-cost 与 transfer cost 均对全部 task 求和。
- 策略分离：
  - `flowcache_always_migrate`：所有 miss 进入 GPU，所有 victim 迁移；
  - `flowcache_selective_migrate_only`：所有 miss 进入 GPU，仅选择性迁移 victim；
  - `flowcache_lossless`：因果 cost-aware GPU admission/bypass + 选择性迁移。
- 选择性 CPU migration：

  \[
  \tilde p_b=
  e^{-\mathrm{age}_b/(\kappa B_G)}
  \frac{w_s s_b+w_f f_b+w_p p_b}{w_s+w_f+w_p}
  \]

  \[
  V_b=\tilde p_b\max(C_b^{pf}-C_b^{H2D},0)
      -C_b^{D2H}-C_b^{hold}
  \]

  仅当 \(V_b>0\) 才迁移；CPU 满时，候选还必须超过最低价值 CPU incumbent。
- Oracle-Cost 启发的 GPU admission/bypass：

  \[
  G_b=\tilde p_b C_b^{pf}
  \]

  \[
  U_{\mathrm{replace}}
  =G_{\mathrm{incoming}}
  +S_{\mathrm{victim}\rightarrow\mathrm{CPU}}^{net},
  \qquad
  U_{\mathrm{keep}}=G_{\mathrm{victim}}
  \]

  满 GPU 上发生 miss 时，仅当
  \(U_{\mathrm{replace}}>U_{\mathrm{keep}}+\delta_G\)
  才把 incoming block 留在 reusable cache；否则当前请求仍正常计算该
  block，但不保留 inactive cache 副本。这样可避免低成本、一次性 cold
  block 为了“强制准入”挤走高价值 hot block。
- 冷启动保护：只有 incoming 的 prefill 成本明显低于 GPU incumbent，
  或 incumbent 已有更强的**已观察复用次数**时，才对 incoming 使用低
  cold-start prior；等成本且等历史证据时保守准入。该规则只读当前与历史，
  类似 TinyLFU doorkeeper，不读取 next-use。它用于防止“少量搬运节省换来
  大量额外 miss”。
- 因果特征：`share_count` 改为 trailing access window 内、截至当前访问的不同 workflow 数；历史访问频率跨物理驱逐保留。固定 horizon 不再把所有老于 1,000 step 的 LRU victim 一律置零。
- 未来信息隔离：只有 `oracle_cost`/Belady 能延迟构建并接收
  `future_accesses`。在线 baseline 若被传入 future index，runner
  fail-closed 报错。相同历史前缀接两种完全不同未来的因果不变性测试已通过。
- 复杂度：transfer 标定查值与静态块信号缓存；GPU victim O(1)，CPU victim 摊还 O(log N)。
- 审计：分别输出 GPU admission 与 CPU migration 的
  candidate/selected/bypassed/rejected、selection rate、restore-per-migration
  与 movement bytes/ms，并显式记录
  `online_feature_scope=current_and_past_only`、
  `future_access_index_used=false`。

当前定向回归测试为 **31/31 通过**（28 项策略/协议测试，加 3 项云端
trace 复用与冻结保护测试）。受控 hot/cold trace
（20,000 accesses，GPU=1 block）的机制回归如下：

| 策略 | hits | migrate / restore | modeled movement | replay wall |
|---|---:|---:|---:|---:|
| Cost-aware GPU bypass + selective migration | 9,999 | 0 / 0 | **0 ms** | 约 0.078 s |
| Always-admit + selective migration | 9,999 | 10,000 / 9,999 | 1,100.943 ms | 约 0.105 s |
| Always-admit + always-migrate | 9,999 | 19,999 / 9,999 | 1,632.890 ms | 约 0.127 s |

因此，原来的 `1,100.9 ms` 不是程序运行了 1.1 秒，而是 20,000 次访问
累计的建模 D2H/H2D 时间；其真实 replay wall 只有约 0.1 秒。新增 GPU
bypass 在不减少 hit 的条件下消除了该受控模式中的全部搬运，并把 replay
wall 在本轮降低约 25%。这仍只是合成机制回归，不是 workload 或论文结果。

另用 200,000-access 等成本循环负对照检查“错误旁路”：完整策略与
Selective-Migrate-Only 都得到 3,213 hits、10,481.116 ms movement，完整
策略 bypass=0。也就是说，成本/历史证据不足时策略回退为保守准入，不再
出现为了减少约 0.38 s 搬运而增加约 32.1 s miss-prefill 的退化。

新增 GPU admission 前，用上游真实 `request_prefixes.jsonl` 重建的
100-request / 4-workflow 局部物理 trace 做过工程 smoke：成本守恒、非负
成本、零 fallback 和复杂度检查均通过，selective
replay/Oracle-Cost wall ratio 为 2.91×。但当时的 CPU-migration-only
策略选择了 23,048/23,052 个候选（99.98%），只减少 0.017% migration，
几乎退化为 always-migrate。该旧 smoke 只能解释本轮为什么增加 incoming
GPU admission，不能证明新策略有效；新策略必须在完整 validation trace
重新运行。

加入“成本偏斜或历史复用证据占优才启用低先验”的因果 doorkeeper 后，
在同一局部物理 trace 上做了第二次只读工程复测（23,107 accesses，
配置 GPU=58 blocks；计入 5% safety margin 后有效 GPU=55、CPU=110）：

| 指标 | 完整 FlowCache | Selective-Migrate-Only | 相对变化 |
|---|---:|---:|---:|
| hit rate | 20.535% | 19.444% | +1.091 个百分点 |
| migrate / restore | 21,996 / 4,745 | 23,048 / 4,493 | migration −4.56% |
| transfer | 1,440.178 ms | 1,481.805 ms | −2.81% |
| miss cost | 1,718,534.9 ms | 1,735,677.0 ms | −0.99% |
| modeled service cost | 1,721,488.6 ms | 1,738,672.3 ms | −0.99% |
| GPU bypass | 1,055 / 18,307 | 0 | 5.76% |
| replay wall | 0.241 s | 0.170 s | +41.8% |

该片段显示新策略不再退化为 always-admit，并且没有用更多 miss 换少搬运；
但 transfer 降幅仍低于预注册的 5% 机制门槛，且局部样本可能与完整
task-grouped split 重叠。因此这些数字只能用于发现实现问题，不能用于
冻结 cost-ratio、选择阈值或形成论文结论。replay wall 增幅也说明必须继续
报告 controller 开销，不能只报告 modeled movement。

## 3. 立即执行：G3-P1

### 3.1 恢复完整 c=4 物理访问流

当前工作区只有完整 `request_prefixes.jsonl`，缺少其派生的
`access_trace_c4.jsonl`。云端入口会直接从 request-prefix trace 生成 c=4
block trace，不再生成约同等体积的中间 `access_trace.jsonl`：

```bash
cd /root/FlowCache
export DATA_DIR=/root/autodl-tmp/flowcache-data
export PREFIXES=$DATA_DIR/request_prefixes.jsonl
export TRACE=$DATA_DIR/access_trace_c4.jsonl
export RUN_DIR=/root/autodl-tmp/flowcache-runs/g3-p1-$(date -u +%Y%m%dT%H%M%SZ)

python .uploads/experiments/g3/run_g3_p1_cloud.py \
  --stage prepare \
  --run-dir "$RUN_DIR" \
  --request-prefixes "$PREFIXES" \
  --trace "$TRACE"
```

builder 只在内存中保留 request offset/调度元数据，第二遍按 offset 直接展开
block；输出使用原子替换并附带 hash/规模/concurrency manifest。不要用
100-request smoke trace 替代正式证据。完整上传、依赖和资源命令见
`.uploads/experiments/g3/CLOUD_RUN.md`。

### 3.2 只在 validation split 选择 admission 参数

按 `task_id` 做稳定哈希分组：20% validation、80% held-out test，
seed=42。同一 task 的 8 个 τ-bench seed 不得跨集合。运行预注册的
54 个参数组合：

```bash
python .uploads/experiments/g3/run_g3_p1_cloud.py \
  --stage tune \
  --run-dir "$RUN_DIR" \
  --request-prefixes "$PREFIXES" \
  --trace "$TRACE"
```

候选只改变四个 admission 参数：

- `minimum_net_benefit_ms ∈ {0, 0.5, 1.0}`；
- `cpu_admission_margin_ms ∈ {0, 0.5}`；
- `gpu_admission_cold_start_cost_ratio ∈ {0.25, 0.5, 0.75}`；
- `expected_cpu_residence_steps ∈ {50, 100, 200}`。

`gpu_admission_margin_ms=0` 与
`gpu_admission_cold_start_prior=0.05`、
`gpu_admission_confidence_scale=1` 在本轮预注册中固定，不用局部 smoke
调整；如 54 组均无有效配置，应先判机制失败原因，而不是在 test 上扩网格。

扫描器复用不随参数变化的结果：GDSF、SizeCost、Always-Migrate 与
Oracle-Cost 各跑一次；Selective-Migrate-Only 按 18 组 CPU 参数各跑一次；
完整 FlowCache 跑 54 组。因而 baseline replay 数由朴素的
`54 × 6 = 324` 降为 `4 + 18 + 54 = 76`，不改变 validation 决策或原始行。
完整 trace 也只解析、task-group split 和因果标注一次，后续候选只读复用；
该缓存不包含 future-access index。

选择目标是在满足以下约束后最小化 validation modeled service cost：

- selection rate 在 1%–99%；
- migration 数相对 always-migrate 至少减少 10%；
- GPU bypass rate 在 1%–99%，防止退化为 always-admit/always-bypass；
- 相对 `flowcache_selective_migrate_only`，transfer 至少减少 5%；
- modeled p95 cache delay 与总 service cost 均不得比
  always-migrate 或 selective-migrate-only 高 5% 以上；
- replay wall 不超过 Oracle-Cost 的 3×；
- transfer 非负、fallback=0。

若 `selection.json` 为 `NO_VALID_CONFIG`，不得查看 test 或继续扩大参数搜索来“追结果”；应重新审查 likelihood proxy/hold cost，或承认该 workload 不支持当前选择机制。

### 3.3 冻结参数后只运行一次 held-out test

云端入口读取 `selection.json`，只允许把 5 个预注册字段（含固定为 0 的
`gpu_admission_margin_ms`）写入新的 `frozen-config.yaml`。若 status 不是
`SELECTED`，冻结器和 held-out test 都会 fail-closed。成功冻结后执行：

```bash
python .uploads/experiments/g3/run_g3_p1_cloud.py \
  --stage test \
  --run-dir "$RUN_DIR" \
  --request-prefixes "$PREFIXES" \
  --trace "$TRACE"
```

固定 cell：2 GiB、concurrency=4。固定对照为 GDSF、SizeCost-LRU、
GPU-only Oracle-Cost（仅离线诊断）、Always-Migrate、
Always-Admit+Selective-Migrate，以及完整 FlowCache。`100000`
只是高于 trace workflow 数的读取上限，不代表新增 100,000 个 workflow。
同一 `RUN_DIR` 若已存在 held-out CSV 或 checker JSON，脚本拒绝第二次测试。

### 3.4 G3-P1 晋级条件

以下全部满足才进入下一阶段：

| 检查 | 条件 | 性质 |
|---|---|---|
| 语义 | 小样本 reference 与优化实现的 hit/miss、GPU/CPU 最终集合一致 | 正确性 |
| 因果性 | 在线策略拒绝 future index；相同历史前缀在不同未来下决策一致 | 正确性 |
| 成本 | transfer cost 全部非负 | 正确性 |
| 守恒 | `sum(task_transfer_ms)` 与 global migrate+restore 相等，误差 ≤ 1e-6 相对量级 | 正确性 |
| 稳定性 | `fallback_count=0` | 正确性 |
| 复杂度 | FlowCache replay wall ≤ 3× Oracle-Cost wall | 内部工程阈值 |
| 选择性 | selection rate ∈ [1%, 99%] | 防止退化为 never/always |
| 搬运收益 | migration 数较 always-migrate 至少下降 10% | 最小机制效应 |
| GPU 准入 | bypass rate ∈ [1%, 99%]，且 accounting 守恒 | 防止退化/漏计 |
| 旁路增益 | transfer 较 selective-migrate-only 至少下降 5% | 隔离新机制贡献 |
| 代价保护 | 相对两项 tiered ablation，modeled p95 与总 service cost 增幅均 ≤ 5% | 防止少搬但多重算 |
| 指标边界 | `ttft_metric_valid=false`、`throughput_metric_valid=false` | 防止过度主张 |
| 状态 | checker 输出 `PASS`，正式 verdict 仍为 `PROTOCOL-INCOMPLETE` | 防止提前 Gate |

参数只能由 validation 决定；test 未通过就记录失败，不得回到 validation 继续选择。复杂度失败先做 profile/heap compaction；选择性或代价保护失败则先判断 proxy 机制是否需要重构。

## 4. G3-P1 之后的真正科学实验

### 4.1 补齐公平 two-tier 策略组

所有方法冻结相同 GPU 与 CPU 容量：

1. GPU-only LRU/GDSF/SizeCost（已有）；
2. Two-tier LRU（待补）；
3. Two-tier GDSF/SizeCost（待补）；
4. Always-Migrate Tiered-LRU（已有独立消融）；
5. Always-Admit + Selective-Migrate（已有独立消融）；
6. Cost-Aware GPU Admission + Selective-Migrate FlowCache（本轮已实现）：

   \[
   \text{admitGPU}(i)\iff
   G_i+S_{v\rightarrow\mathrm{CPU}}^{net}>G_v+\delta_G
   \]

   \[
   \text{migrate}(b)\iff
   \widehat S_{b,\mathrm{CPU}}
   -C^{place}_{b}
   -C^{hold}_{b}>0
   \]

7. Two-tier Oracle-Cost（待补）。

需要报告 CPU tier 带来的纯容量收益，以及 selective decision 在相同总资源上的增量收益。否则无法区分“方法更聪明”与“多给了一层内存”。

### 4.2 主 cell closed-loop serving

只在一个主 cell 先运行真实服务：

- 记录 request arrival、service start、first token、completion；
- 测 TTFT/JCT p50/p95/p99、queueing、achieved throughput、SLO goodput；
- 用 CUDA event 或后端等价机制测 H2D/D2H，并报告是否与 prefill/compute 重叠；
- 报告 GPU allocated/reserved、CPU pinned bytes、PCIe bytes、controller p50/p95；
- 用相同 request 顺序、预算、模型、dtype 和并发比较全部策略。

只有 closed-loop 数据才能应用 G3 的正式门槛：p95 TTFT 改善约 15%、throughput 下降不超过约 5%、并显著优于 two-tier GDSF/SizeCost。

### 4.3 决策顺序

```text
G3-P1 validation 冻结 admission 参数
        ↓
held-out test 单 cell选择性/复杂度检查
        ↓ PASS
补齐公平 two-tier baselines
        ↓
主 cell closed-loop TTFT / throughput / goodput
        ↓ PASS
9-cell 网格与 workflow-level bootstrap
        ↓
G3 GO / NO-GO
        ↓ GO
G4 Quantization
```

## 5. 后续研究方向

1. **联合校准 GPU admission 与 CPU migration**：159,210 / 641,863
   ≈ 24.8% 的旧 restore-to-migrate 比例提示大量搬运未回收价值；下一步
   分别报告 GPU bypass rate、CPU selection rate、movement reduction、
   restore yield、额外 miss cost 与 policy regret，而不是只看 hit rate。
2. **同资源、bypass-aware two-tier oracle**：建立同时比较 incoming、
   GPU incumbent、CPU incumbent 与 evict 的离线诊断。现有
   Oracle-Cost 的 cost/distance 比例是 lookahead heuristic，不应再称为
   严格最优上界。
3. **异步迁移与预取**：在真实后端研究 H2D/D2H 与 tool wait、prefill、decode 的重叠；同步成本相加只是保守模型。
4. **反抖动机制**：加入 minimum residency、hysteresis 和带宽预算，控制 GPU↔CPU 往返。
5. **再进入量化**：只有 G3 无损路径在真实 TTFT/goodput 上成立，才做 G4；否则量化会把协议问题和质量风险混在一起。
