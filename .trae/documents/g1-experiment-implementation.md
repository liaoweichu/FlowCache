# G1 实验代码实现计划

> **目标**：实现 FlowCache G1（Opportunity）实验代码，覆盖 W3–W7 全流程：trace 录制 → 画像 → 策略对比 → 判定报告。
> **上游设计**：`experiments/experiment-designs.md` G1/E1 章节（v0.3.1）、`.trae/specs/experiment-scope-redesign/spec.md` v0.3
> **创建日期**：2026-07-25
> **状态**：planning
>
> **⚠️ v0.5（2026-07-26） supersession 注**：本文档中所有 BFCL 作为数据集的引用（"τ-bench + BFCL 800"、"7,720 episodes"、BFCL 6,400 / 8 seeds、BFCL 集成方式等）**已全部作废**。当前设计为单数据集 τ-bench 1,320 episodes（165 tasks × 8 seeds），BFCL 不再作为数据集（详见 `experiments/experiment-designs.md` v0.5 注与 `IDEA.rewritten.md` §6.1）。本文档保留作为历史决策记录，不再代表当前计划。BFCL 相关代码（`bfcl_adapter.py`）作为 disabled adapter 保留，`config.yaml` 已设为 τ-bench only。

---

## 1. 当前状态分析

### 1.1 已有资产（可直接复用）

| 组件 | 路径 | G1 用途 |
|---|---|---|
| `Backend.slice_kv_into_blocks` / `restore_kv_from_blocks` | [experiments/g0/backend.py](file:///d:/00MyProject/Prefix%20Caching/experiments/g0/backend.py) | KV 拦截/恢复（replay 原语） |
| `Backend.safe_forward` + 显存监控 | 同上 | 1320+6400 episodes 长跑 OOM 防护 |
| `Backend.get_model_info` | 同上 | I_b 元数据（m/r/tau/c） |
| `compute_block_hash`（8 元组版） | [experiments/g0/block_index.py](file:///d:/00MyProject/Prefix%20Caching/experiments/g0/block_index.py) | 替代 e1 简化版（4 元组） |
| `compute_template_hash` / `compute_config_hash` | 同上 | I_b 元数据 |
| `LRUCache` / `GDSFCache` / `BeladyOracle` | [experiments/e1/compare_oracle.py](file:///d:/00MyProject/Prefix%20Caching/experiments/e1/compare_oracle.py) | 3/7 策略已实现 |
| `compute_workflow_structure` / `compute_exact_prefix_overlap` / `compute_next_use_distance` / `compute_working_set` | [experiments/e1/characterize_workload.py](file:///d:/00MyProject/Prefix%20Caching/experiments/e1/characterize_workload.py) | 4 类画像指标 |
| `trace_utils.load_all_trajectories` | [experiments/e1/trace_utils.py](file:///d:/00MyProject/Prefix%20Caching/experiments/e1/trace_utils.py) | trace 加载 |
| `plot_oracle_comparison` / `plot_overlap_histogram` 等 | [experiments/e1/plot_characterization.py](file:///d:/00MyProject/Prefix%20Caching/experiments/e1/plot_characterization.py) | 图表生成 |

### 1.2 主要缺口

1. **多 seed 支持**：`record_trajectories.py` 当前 `do_sample=False`，无 seed 参数
2. **BFCL v3 集成**：完全未实现，仅支持 τ-bench
3. **真实 τ-bench backend**：当前 `_simulate_tool_result` / `_simulate_user_response` 是 mock，需对接 `tau-bench` pip 包
4. **4 个策略缺失**：APC-LRU、SizeCost、Oracle-Cost、KVFlow/PBKV
5. **replay 协议不全**：缺 arrival_time、3 replay seeds、p95 TTFT、miss_cost
6. **画像扩展**：`compute_working_set` 是累加不淘汰，需补滑动窗口 H=1000；`compute_exact_prefix_overlap` O(n²) 在 n=7720 不可行
7. **trace 格式**：缺 seed/dataset/bfcl_subset/model_id/revision/template_hash/config_hash/per-step arrival_time_ms 字段
8. **block identity 一致性**：e1 简化版 hash 与 G0 8 元组版不一致，需统一

### 1.3 关键约束

- **硬件**：RTX 4090D 24GB，~26 GPU 小时 Tier-1 录制预算（spec v0.3 §8）
- **样本量**：τ-bench 1,320（165 × 8 seeds）+ BFCL 6,400（800 × 8 seeds）= **7,720 episodes**
- **时间窗**：W3–W7（5 周）
- **无损路径**：G1 不启用量化（G1.1.1）
- **open-loop replay**：冻结 token IDs/工具结果/到达时间（G1.6）

---

## 2. 已确认的关键决策

| 决策点 | 选择 | 依据 |
|---|---|---|
| BFCL 多 seed | **8 seeds**（model decode seed，do_sample=True, temperature=0.7） | 与 τ-bench 对齐，总 episodes 7720；算力 ~50 GPU 小时（需从 ~26 上调） |
| BFCL 集成方式 | **pip install bfcl-eval** + `--include-input-log` | 复用官方 8 个 sim 工具类与状态验证；先做依赖矩阵测试 |
| Closest baseline | **KVFlow 和 PBKV 都做 G1.4.1 检查清单** | 选可忠实运行的那个；若都不可则标 *-inspired |
| block identity | **统一用 G0 版 8 元组** | 废弃 e1 简化版；现有 e1 traces 作废重录 |
| seeds 语义 | τ-bench = user simulator seed；BFCL = model decode seed | 报告需明确披露此差异 |
| Ch.5 压力数据集精简 | **3 → 1**（保留 STB 500，删 SWE 200 + Toolathlon 200） | 节省 ~4.5 GPU 小时；200 样本单独成章证据力弱；与 C1–C3 主线关联弱（见 trim-dataset-portfolio spec） |

### 2.1 算力预算上调

spec v0.3 §8 原 ~26 GPU 小时基于 τ-bench 1320 + BFCL 800（单 seed）。用户选定 BFCL 也做 8 seeds 后：

| 项 | 原预算 | 新预算 |
|---|---|---|
| τ-bench 录制 | 1320 × ~30s = 11 GPU 小时 | 11 GPU 小时（不变） |
| BFCL 录制 | 800 × ~20s = 4.5 GPU 小时 | 6400 × ~20s = **36 GPU 小时** |
| Tier-1 总录制 | ~26 GPU 小时 | **~50 GPU 小时** |

**应对**：W3–W5 录制窗口延长；BFCL 单 episode 较短（1–7 user turn），实际可能 < 20s/episode；优先录制 τ-bench，BFCL 可分批录制。

---

## 3. 实施步骤（按周次）

### Step 1: 数据集适配器（W3，~3 天）

#### 1.1 新增 `experiments/e1/taubench_adapter.py`

**职责**：替换现有 `_simulate_tool_result` / `_simulate_user_response`，对接 `tau-bench` pip 包的真实 backend。

```python
class TauBenchAdapter:
    def __init__(self, domain: str, seed: int): ...
    def load_tasks(self) -> List[Task]:  # 165 任务全量
    def init_env(self, task: Task) -> Env:  # retail/airline env + llm_user simulator
    def step(self, action) -> Tuple[obs, reward, done, info]:  # 真实工具执行
    def get_system_policy(self) -> str:  # 真实 policy（替代 _get_domain_policy 硬编码）
```

**关键点**：
- `pip install tau-bench`，import `tau_bench.envs.tool.retail` / `tau_bench.envs.tool.airline`
- `llm_user` 模拟器用 `gpt-4o-mini` 或本地 Qwen2.5-7B-Instruct（显存允许时）
- seed 注入 `llm_user` 的 temperature + seed（与原论文 pass^k 对齐）
- 165 任务全量加载（115 retail + 50 airline），不再用 `max_workflows=80` 截断

#### 1.2 新增 `experiments/e1/bfcl_adapter.py`

**职责**：加载 BFCL v3 multi-turn 4 子集 × 200 = 800 episodes，复用官方 backend。

```python
class BFCLAdapter:
    def __init__(self, subset: str, seed: int): ...  # subset ∈ {multi_turn_base, miss_func, miss_param, long_context}
    def load_episodes(self) -> List[Episode]:  # 200 episodes/子集
    def init_backend(self, initial_config: Dict) -> SimBackend:  # 8 个 sim 类之一，copy.deepcopy 快照
    def step(self, action) -> Tuple[obs, reward, done, info]:  # 真实工具执行 + 状态验证
    def get_tool_schema(self) -> str:  # function 字段 → system prompt 的 tool schema 部分
```

**关键点**：
- `pip install bfcl-eval`，先做依赖矩阵测试（transformers 版本兼容性）
- 用 `bfcl generate --test-category {subset} --include-input-log` 或直接读 `bfcl_eval/data/BFCL_v3_*.json`
- scripted user turns 从 `question` 字段读，无 LLM 模拟器
- seed 注入 model.generate（do_sample=True, temperature=0.7, seed=k）
- 8 个 sim 类：Vehicle Control / Trading Bots / Travel Booking / Gorilla File System / Message API / Twitter API / Ticket API / Math API

#### 1.3 验证

- τ-bench：跑 5 episodes smoke test，确认 `llm_user` 能正常多轮对话
- BFCL：跑 5 episodes smoke test，确认 backend 状态验证通过
- 依赖矩阵：`pip list` 检查 transformers/torch/bfcl-eval/tau-bench 版本兼容性

---

### Step 2: 录制管线改造（W3–W4，~4 天）

#### 2.1 修改 `experiments/e1/record_trajectories.py`

**改动点**：

| 位置 | 改动 |
|---|---|
| argparse / config | 新增 `--seed`、`--dataset`（tau-bench/bfcl_v3）、`--bfcl-subset` |
| `_init_model` | 保留 `do_sample=False` 用于 τ-bench（greedy，seed 注入 user simulator）；BFCL 分支用 `do_sample=True, temperature=0.7, seed=k` |
| `run_workflow` | 根据 dataset 字段选择 adapter（TauBenchAdapter / BFCLAdapter）；替换 `_simulate_tool_result` / `_simulate_user_response` 为真实 adapter.step |
| `record_all` | 外层加 `for seed in seeds:` 循环；输出文件名 `{task_id}_seed{seed}.json`；输出目录 `traces/bf16/{tau_bench,bfcl_v3}/` |
| `meta` 字段 | 新增 `seed`、`dataset`、`bfcl_subset`、`group_id`、`pass_k`、`model_id`、`revision`、`template_hash`、`config_hash`、`adapter_id` |
| `step` 字段 | 新增 `arrival_time_ms`（相对 workflow 起点）、`tool_wait_ms`（tool_call 发出到 tool_result 返回的 wall-clock） |
| block identity | 改用 `g0/block_index.py` 的 8 元组 `compute_block_hash`，废弃 e1 简化版 |
| checkpoint/resume | 每 N 个 workflow 落盘；启动时扫描已存在文件 skip；7720 episodes 长跑必备 |
| OOM 防护 | 复用 `Backend.safe_forward` + `assert_memory_available`；OOM 时 skip 当前 episode 并记录 |

**关键实现**：
```python
# record_all 改造伪代码
def record_all(self, subset_path=None):
    datasets = self._config["workload"]["datasets"]  # ["tau-bench", "bfcl_v3"]
    seeds = self._config["workload"]["seeds"]  # [42, 123, ..., 192021] (8 个)
    for dataset in datasets:
        for seed in seeds:
            adapter = self._init_adapter(dataset, seed)
            for task in adapter.load_tasks():
                out_path = self._output_dir / dataset / f"{task.id}_seed{seed}.json"
                if out_path.exists() and self._config.get("resume", True):
                    continue  # checkpoint resume
                try:
                    self._run_workflow_with_backend(task, adapter, seed, out_path)
                except OOMError:
                    self._log_oom_skip(task, seed)
```

#### 2.2 修改 `experiments/e1/trace_utils.py`

- `compute_block_hash` 改为从 `g0/block_index.py` import（保持唯一实现）
- `load_trajectory` 校验 meta 新字段（seed/dataset/model_id/revision 等），缺失时 warning

#### 2.3 修改 `experiments/e1/config.yaml`

```yaml
model:
  name: "/autodl-pub/models/Qwen2.5-7B-Instruct"
  dtype: "bfloat16"
  trust_remote_code: true
  device_map: "auto"

workload:
  datasets: ["tau-bench", "bfcl_v3"]
  seeds: [42, 123, 456, 789, 101112, 131415, 161718, 192021]  # 8 seeds
  tau_bench:
    tasks: 165  # 全量（115 retail + 50 airline）
    user_simulator: "llm_user"
  bfcl_v3:
    subsets: ["multi_turn_base", "miss_func", "miss_param", "long_context"]
    per_subset: 200
    decode_mode: "sampling"  # do_sample=True, temperature=0.7
  concurrency: 4  # 录制时单 GPU 串行，replay 时用

cache:
  block_size: 16
  kv_budgets: [0.10, 0.25, 0.50, 1.00]
  scheduling_window_h: 1000

replay:
  arrival_process: "poisson"
  arrival_lambda: 4
  arrival_sources: ["burst_gpt", "poisson"]
  num_replay_seeds: 3
  tool_wait_tolerance: 1.0

output:
  trace_dir: "traces/bf16"
  trace_subdirs: ["tau_bench", "bfcl_v3"]
  report_dir: "outputs"
  figure_dir: "figures"
  report_format: ["json", "markdown"]
  resume: true  # checkpoint/resume
```

#### 2.4 验证

- 单 seed × 单 dataset 跑 10 episodes，检查 trace 文件格式（新字段齐全）
- `verify_parent_chain` 校验 block 父链
- `Backend.restore_kv_from_blocks` 用 trace 中的 block_assignments 还原 KV，与录制时 past_kv 比对（bit-identical）

---

### Step 3: 全量录制（W5，~50 GPU 小时）

#### 3.1 执行

```bash
# τ-bench 1320 episodes（165 × 8 seeds），~11 GPU 小时
python experiments/e1/record_trajectories.py --config experiments/e1/config.yaml --dataset tau-bench

# BFCL 6400 episodes（800 × 8 seeds），~36 GPU 小时
python experiments/e1/record_trajectories.py --config experiments/e1/config.yaml --dataset bfcl_v3
```

#### 3.2 产物

- `experiments/e1/traces/bf16/tau_bench/{task_id}_seed{seed}.json` × 1320
- `experiments/e1/traces/bf16/bfcl_v3/{subset}_{episode_id}_seed{seed}.json` × 6400
- `experiments/e1/traces/bf16/_recording_report.json`（含 OOM skip 记录）

#### 3.3 验证

- 文件数 = 1320 + 6400 = 7720
- 抽样 10 个 trace，`load_trajectory` + `verify_parent_chain` 通过
- 抽样 5 个 trace，`Backend.restore_kv_from_blocks` 还原 KV 与录制时 bit-identical

---

### Step 4: 画像扩展（W6 上半，~2 天）

#### 4.1 修改 `experiments/e1/characterize_workload.py`

**改动点**：

| 函数 | 改动 |
|---|---|
| `compute_workflow_structure` | 按 dataset 分组输出；新增 per-step `tool_wait_ms` 统计 |
| `compute_exact_prefix_overlap` | **优化 O(n²) → O(n)**：用 block hash 桶（Dict[hash, List[workflow_id]]）替代两两配对；7720 episodes 不可两两配对 |
| `compute_next_use_distance` | 补全 next-use sequence（不只第二次访问）；按 dataset 分组 |
| `compute_working_set` | **补滑动窗口 H=1000**：用 deque 维护窗口内 unique block KV 字节峰值；替代当前累加不淘汰 |
| `_write_markdown_report` | 按 dataset 分节输出；新增 total LLM calls、平均 turn/episode、pass^k（k∈{1,2,4,8}） |

**关键实现**（滑动窗口）：
```python
def compute_working_set(trajectories, block_size, window_h=1000):
    # 把所有 step 展平为全局 step 序列
    global_steps = []
    for traj in trajectories:
        for step in traj["steps"]:
            for block in step["block_assignments"]:
                global_steps.append((step["step_id"], block["block_hash"], traj["meta"]["workflow_id"]))
    
    # 滑动窗口维护 unique block KV 字节
    window = deque()  # (global_step, block_hash)
    block_in_window = defaultdict(int)  # block_hash -> count
    peak = 0
    for gs, bh, wf in global_steps:
        window.append((gs, bh))
        block_in_window[bh] += 1
        while window and window[0][0] <= gs - window_h:
            old_gs, old_bh = window.popleft()
            block_in_window[old_bh] -= 1
            if block_in_window[old_bh] == 0:
                del block_in_window[old_bh]
        peak = max(peak, len(block_in_window) * per_block_kv_bytes)
    return {"working_set_size": peak, ...}
```

#### 4.2 产物

- `experiments/e1/outputs/e1-characterization.json`（按 dataset 分组）
- `experiments/e1/outputs/e1-report.md`（G1.11.1 表 G1-3 locality 画像摘要）

#### 4.3 验证

- 画像指标分布合理（overlap > 0、next-use distance 有方差）
- τ-bench vs BFCL 对比图清晰
- pass^k 统计与原论文量级一致

---

### Step 5: 策略实现（W6 下半，~3 天）

#### 5.1 新增 `experiments/g1/` 目录结构

```
experiments/g1/
├── __init__.py
├── config.yaml
├── strategies/
│   ├── __init__.py
│   ├── apc_lru.py          # 新增
│   ├── size_cost.py         # 新增
│   ├── oracle_cost.py       # 新增
│   └── closest_baseline.py  # 新增（KVFlow + PBKV）
├── cost_model.py             # 新增（C^res 成本模型）
├── replay_driver.py          # 新增（open-loop replay 驱动器）
├── baseline_comparability.py # 新增（G1.4.1 5 项检查清单）
└── verdict.py                # 新增（G1 判定报告）
```

#### 5.2 新增 `experiments/g1/strategies/apc_lru.py`

**APC-LRU**：工程 baseline，block 级 LRU + prefix tree。

```python
class APCLRU:
    def __init__(self, capacity_blocks: int, block_size: int): ...
    def access(self, block_hash: str, parent_hash: str) -> bool:  # hit/miss
    def evict(self) -> Optional[str]:  # LRU evict，维护 prefix tree 一致性
```

**实现要点**：参考 vLLM `LRUCacheEvictionPolicy`；block 级 LRU + prefix tree（子 block 在父 block evict 时级联失效）。

#### 5.3 新增 `experiments/g1/strategies/size_cost.py`

**SizeCost**：强启发式，priority = age + size + measured_recompute_cost。

```python
class SizeCost:
    def __init__(self, capacity_blocks: int, cost_model: CostModel): ...
    def access(self, block_hash: str, token_count: int) -> bool:
    def evict(self) -> Optional[str]:  # min priority evict
```

**priority 公式**（IDEA §4.3 第一档）：
```
priority = α * age + β * size + γ * recompute_cost
evict: min(priority)
```
- `recompute_cost` 由 `cost_model.estimate(block_hash)` 提供
- α/β/γ 默认 1.0/1.0/1.0，pilot 后标定

#### 5.4 新增 `experiments/g1/strategies/oracle_cost.py`

**Oracle-Cost**：上界，未来已知 + 成本模型。

```python
class OracleCost:
    def __init__(self, capacity_blocks: int, future_accesses: Dict[str, List[int]], cost_model: CostModel): ...
    def access(self, block_hash: str, current_step: int) -> bool:
    def evict(self, current_step: int) -> Optional[str]:  # min(future_value, recompute_cost) evict
```

**evict 决策**：
```
for block in cache:
    next_use = next_access_after(block, current_step)  # ∞ if never
    future_value = 1 / (next_use - current_step)  # 越近用越大
    recompute_cost = cost_model.estimate(block)
    score = min(future_value, recompute_cost)
evict: min(score)
```

#### 5.5 新增 `experiments/g1/strategies/closest_baseline.py`

**KVFlow + PBKV**：先做 G1.4.1 5 项检查清单，可忠实运行则移植，否则标 `*-inspired`。

```python
class KVFlowAdapter:
    def __init__(self, capacity_blocks: int): ...
    def check_comparability(self) -> Dict:  # G1.4.1 5 项
    def access(self, ...) -> bool:  # 若可忠实运行则调用官方逻辑
    def is_faithful(self) -> bool:  # True=faithful, False=*-inspired

class PBKVAdapter:
    # 同上
```

**G1.4.1 5 项检查清单**（对 KVFlow 和 PBKV 各做一次）：
1. 官方代码/协议可获得性
2. 所需引擎钩子（block index / eviction hook / prefetch hook）本后端是否具备
3. 其缓存语义是否与本研究 exact-prefix 语义一致
4. 其特征是否违反本研究禁止特征清单（未来信息泄漏检查）
5. 在本 replay 协议下可忠实运行的 trace 覆盖率

#### 5.6 新增 `experiments/g1/cost_model.py`

**C^res 成本模型**（IDEA §2.1 实测建模）。

```python
class CostModel:
    def __init__(self, backend: Backend): ...
    def calibrate(self):  # 实测 α/β/γ（不同 batch_size、context_len、decode_len）
    def estimate(self, block_hash: str) -> float:  # 返回 recompute cost (ms)
```

**标定流程**：
- 在 calibration 阶段用 `Backend.safe_forward` 实测不同 context_len 的 prefill_ms
- 拟合 `C^res = α * num_prefill_tokens + β * num_decode_tokens + γ * KV_size`
- 标定结果存 `experiments/g1/cost_model_calibration.json`

#### 5.7 验证

- 4 个新策略单元测试（access/evict 逻辑正确）
- `cost_model.calibrate()` 在 5 个 context_len 上拟合 R² > 0.95
- KVFlow/PBKV 的 `check_comparability()` 输出 5 项清单

---

### Step 6: Replay Driver 与全网格运行（W7 上半，~2 天）

#### 6.1 新增 `experiments/g1/replay_driver.py`

**职责**：open-loop replay 驱动器，按 trace 顺序驱动策略，统计 miss_cost、p95 TTFT、hit_rate。

```python
class ReplayDriver:
    def __init__(self, traces: List[Trajectory], strategies: List[Strategy], 
                 cost_model: CostModel, arrival_process: str): ...
    def run(self, budget: float, replay_seed: int) -> Dict:
        # 1. 按 arrival_process 生成到达时间（Poisson λ=4 或 BurstGPT 窗口）
        # 2. 按 trace 顺序驱动：for step in trace: for block in step.block_assignments: strategy.access(block)
        # 3. 统计：hits, misses, evictions, miss_cost_ms, p95_ttft_ms, hit_rate
        # 4. 返回 per-strategy 结果
```

**关键实现**：
- 到达时间：BurstGPT 真实窗口为主证据，Poisson λ=4 为建模参照（报告拟合优度）
- 3 个 replay 种子（到达时间扰动）：每个（策略 × 预算 × 数据集）单元运行 3 次
- miss_cost：block 未命中时重算的 prefill_ms（由 cost_model 估计）
- p95 TTFT：每个请求的首 token 延迟（含 miss 重算 + hit 复用）

#### 6.2 修改 `experiments/e1/compare_oracle.py`

- 接入 4 个新策略（APC-LRU / SizeCost / Oracle-Cost / KVFlow / PBKV）
- 替换 `build_access_trace` 为支持 arrival_time 的版本
- 输出 CSV（`experiments/g1/results/*.csv`）+ JSON

#### 6.3 全网格运行

```bash
# 7 策略 × 4 预算 × 2 数据集 × 3 replay seeds = 168 runs
python experiments/g1/replay_driver.py --config experiments/g1/config.yaml
```

**网格**：
- 策略：APC-LRU, LRU, GDSF, SizeCost, Oracle-Belady, Oracle-Cost, KVFlow†/PBKV†
- 预算：10%, 25%, 50%, 100%
- 数据集：τ-bench (1320), BFCL (6400)
- replay seeds：3 个

#### 6.4 产物

- `experiments/g1/results/headroom_tau_bench.csv`
- `experiments/g1/results/headroom_bfcl.csv`
- `experiments/g1/results/baseline-comparability.md`（G1.4.1 检查清单结果）

#### 6.5 验证

- 168 runs 全部完成（无 crash）
- Oracle-Cost ≥ Oracle-Belady ≥ 简单策略（合理性检查）
- p95 TTFT 与 miss_cost 单调相关

---

### Step 7: 判定报告与图表（W7 下半，~1 天）

#### 7.1 新增 `experiments/g1/verdict.py`

**职责**：生成 G1 判定报告。

```python
def check_condition_1_headroom(results: Dict) -> Dict:
    # oracle vs 最佳简单策略的 miss-cost 或 p95 TTFT 改进 ≥ 10%
    
def check_condition_2_comparability(comparability: Dict) -> Dict:
    # ≥1 个 PBKV/KVFlow 在公平协议下忠实运行
    
def generate_verdict(results: Dict, comparability: Dict) -> Dict:
    # 两项同时通过 → G1 passed
```

#### 7.2 修改 `experiments/e1/plot_characterization.py`

- `plot_oracle_comparison` 扩展到 7 策略线
- 新增 `plot_headroom_main`（G1.11.1 表 G1-1/G1-2 的可视化）
- 新增 `plot_pass_k`（τ-bench pass^k 曲线）

#### 7.3 产物

- `experiments/g1/g1-verdict.md`（G1 判定报告，G1.11.1 表 G1-1/G1-2/G1-3 填充）
- `figures/g1-headroom.png`（headroom 主图）
- `figures/g1-pass-k.png`（τ-bench pass^k 曲线）
- `experiments/g1/baseline-comparability.md`（closest baseline 可比性记录）

#### 7.4 验证

- G1 判定报告含两项判定结果 + 证据
- 表 G1-1/G1-2/G1-3 全部填充（无 TBD）
- 图表清晰展示 headroom 与 pass^k

---

## 4. 文件清单

### 4.1 新增文件（11 个）

| 文件 | 行数估计 | 职责 |
|---|---|---|
| `experiments/e1/taubench_adapter.py` | ~200 | τ-bench 真实 backend 集成 |
| `experiments/e1/bfcl_adapter.py` | ~250 | BFCL v3 multi-turn 数据加载 + backend |
| `experiments/g1/__init__.py` | 1 | 包标记 |
| `experiments/g1/config.yaml` | ~50 | G1 配置 |
| `experiments/g1/strategies/apc_lru.py` | ~100 | APC-LRU 策略 |
| `experiments/g1/strategies/size_cost.py` | ~100 | SizeCost 策略 |
| `experiments/g1/strategies/oracle_cost.py` | ~120 | Oracle-Cost 策略 |
| `experiments/g1/strategies/closest_baseline.py` | ~200 | KVFlow + PBKV 适配 + 可比性检查 |
| `experiments/g1/cost_model.py` | ~150 | C^res 成本模型 + 标定 |
| `experiments/g1/replay_driver.py` | ~250 | open-loop replay 驱动器 |
| `experiments/g1/verdict.py` | ~150 | G1 判定报告生成 |

### 4.2 修改文件（6 个）

| 文件 | 改动量 | 改动点 |
|---|---|---|
| `experiments/e1/record_trajectories.py` | 大 | seed 循环、dataset 分支、真实 backend、trace.meta 字段扩展、block identity 升级、checkpoint/resume |
| `experiments/e1/trace_utils.py` | 中 | `compute_block_hash` 改用 G0 版；`load_trajectory` 校验新字段 |
| `experiments/e1/characterize_workload.py` | 中 | 滑动窗口、O(n²) 优化、按 dataset 分组、pass^k |
| `experiments/e1/compare_oracle.py` | 中 | 接入 4 新策略、arrival_time、3 replay seeds、p95 TTFT |
| `experiments/e1/config.yaml` | 中 | 多数据集、seeds=8、trace_dir 分层、resume |
| `experiments/e1/plot_characterization.py` | 小 | 7 策略线、headroom 主图、pass^k 图 |

---

## 5. 假设与决策

### 5.1 假设

1. `tau-bench` pip 包可正常安装并提供 `llm_user` 模拟器接口
2. `bfcl-eval` pip 包与 transformers 版本兼容（先做依赖矩阵测试）
3. 4090D 24GB 可容纳 Qwen2.5-7B-Instruct BF16（~15GB）+ KV cache + 8 个 sim backend
4. ~50 GPU 小时录制预算可在 W3–W5 窗口内完成（含 checkpoint/resume）
5. G1.4.1 检查清单可在 W6 下半完成（KVFlow/PBKV 官方代码可获得）

### 5.2 决策

1. **block identity 统一**：G1 用 G0 版 8 元组 `compute_block_hash`，废弃 e1 简化版；现有 e1 traces 作废重录
2. **BFCL seed 语义**：BFCL 的 8 seeds 是 model decode seed（do_sample=True, temperature=0.7），与 τ-bench 的 user simulator seed 不同，报告需明确披露
3. **closest baseline 顺序**：W6 下半同时检查 KVFlow 和 PBKV，选可忠实运行的那个；若都不可则标 *-inspired
4. **replay 种子**：3 个 replay 种子（到达时间扰动），与录制 seeds（8 个）独立
5. **算力预算上调**：BFCL 8 seeds 后总录制 ~50 GPU 小时（原 ~26），W3–W5 窗口延长

---

## 6. 验证步骤

### 6.1 单元验证（每步完成后）

| 步骤 | 验证项 | 通过标准 |
|---|---|---|
| Step 1 | τ-bench/BFCL adapter smoke test | 5 episodes 跑通，backend 状态验证通过 |
| Step 2 | trace 文件格式 | 新字段齐全（seed/dataset/model_id/revision/template_hash/config_hash/arrival_time_ms/tool_wait_ms） |
| Step 2 | block identity 一致性 | `Backend.restore_kv_from_blocks` 还原 KV 与录制时 bit-identical |
| Step 3 | 全量录制 | 7720 文件齐全，`verify_parent_chain` 全通过 |
| Step 4 | 画像合理性 | overlap > 0、next-use distance 有方差、pass^k 与原论文量级一致 |
| Step 5 | 策略单元测试 | 4 新策略 access/evict 逻辑正确；cost_model R² > 0.95 |
| Step 6 | 168 runs 全完成 | 无 crash；Oracle-Cost ≥ Oracle-Belady ≥ 简单策略 |
| Step 7 | G1 判定报告 | 两项判定结果 + 证据；表 G1-1/G1-2/G1-3 全填充 |

### 6.2 集成验证（W7 末）

- G1 判定报告 `experiments/g1/g1-verdict.md` 完整
- `ccfa.yaml` 更新：G1 status → passed/failed，stage → g1_passed/转路线 B
- 所有产物路径与 G1.11 一致

---

## 7. 风险与应对

| 风险 | 概率 | 应对 |
|---|---|---|
| `bfcl-eval` 与 transformers 版本冲突 | 中 | W3 先做依赖矩阵测试；若冲突则切拷贝源码方案 |
| 7720 episodes 录制超预算 | 中 | checkpoint/resume；优先 τ-bench，BFCL 分批；BFCL 单 episode 较短可能 < 20s |
| KVFlow/PBKV 都不可忠实运行 | 中 | 标 *-inspired，记录 ≤5 项不兼容原因；触发 IDEA §11 风险条目 |
| `compute_exact_prefix_overlap` O(n²) 优化后仍慢 | 低 | 进一步按 block hash 桶 + 采样（如 1000 sample 估算） |
| BFCL 8 seeds 显存不足 | 低 | BFCL 单 episode turn 数少（1–7），显存压力小于 τ-bench；必要时降到 3 seeds |

---

## 8. 执行顺序甘特图

```
W3  │─── Step 1: 数据集适配器 ───│
W4  │─── Step 2: 录制管线改造 ───│
W5  │────────── Step 3: 全量录制（~50 GPU 小时）──────────│
W6  │─── Step 4: 画像扩展 ───│─── Step 5: 策略实现 ───│
W7  │─── Step 6: Replay + 全网格 ───│─── Step 7: 判定报告 ───│
```

---

## 9. 后续动作

1. 本计划获用户批准后，调用 writing-plans skill 创建详细 tasks.md
2. W3 开始执行 Step 1
3. 每步完成后更新 `ccfa.yaml` 的 stage.next_week 与 todo 状态
4. G1 判定通过后，更新 `ccfa.yaml`：G1 status → passed，stage → g1_passed，gate → G2
