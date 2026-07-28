# G3-P1 云端执行手册

本流程只运行 G3-P1 open-loop 工程诊断，不产生真实 TTFT、throughput 或
科学 GO/NO-GO。推荐按 `prepare → tune → test` 三阶段执行，避免在
held-out test 上调参。

## 1. 上传到云端

若云端尚无项目，在本地 Windows PowerShell 的项目根目录打包并上传：

```powershell
Set-Location "D:\00MyProject\Prefix Caching FOR gpt"
tar -czf flowcache-g3-cloud.tgz .uploads
Get-FileHash .\flowcache-g3-cloud.tgz -Algorithm SHA256
scp .\flowcache-g3-cloud.tgz root@<服务器IP>:/root/
```

服务器端解包，并把大 trace 放到数据盘：

```bash
mkdir -p /root/FlowCache
mkdir -p /root/autodl-tmp/flowcache-data
tar -xzf /root/flowcache-g3-cloud.tgz -C /root/FlowCache

mv /root/FlowCache/.uploads/experiments/g1prime/physical_traces/request_prefixes.jsonl \
  /root/autodl-tmp/flowcache-data/request_prefixes.jsonl
```

若项目和 `request_prefixes.jsonl` 已在服务器上，只需把后面的
`PREFIXES` 改成实际绝对路径，不必重新上传。

## 2. 云端环境

在项目根目录执行：

```bash
cd /root/FlowCache

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  -r .uploads/experiments/g3/requirements-cloud.txt

python --version
python -c "import yaml; print(yaml.__version__)"
```

该 open-loop replay 不使用 GPU；云端瓶颈是 CPU、RAM 和本地 SSD。
`run_g3_p1_cloud.py` 会根据 `request_prefixes.jsonl` 样本估计完整 trace
大小，并以约 `trace_size × 6` 做保守内存预检。

当前 477,993,937-byte 输入的本地预检估计 c=4 trace 约 1.30 GB、保守
replay 内存约 7.82 GB。建议至少 16 GB RAM、5 GB 可用 SSD；若同时运行
其他任务，应使用更高内存。增加 GPU 不会加速这一 open-loop 阶段。

## 3. 推荐：分阶段运行

使用持久化终端，避免 SSH 断线：

```bash
tmux new -s flowcache-g3
cd /root/FlowCache
source .venv/bin/activate

export DATA_DIR="/root/autodl-tmp/flowcache-data"
export PREFIXES="$DATA_DIR/request_prefixes.jsonl"
export TRACE="$DATA_DIR/access_trace_c4.jsonl"
export RUN_DIR="/root/autodl-tmp/flowcache-runs/g3-p1-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RUN_DIR"
test -s "$PREFIXES"
echo "$RUN_DIR"
```

### 阶段 A：预检、直接生成 c=4 trace、回归测试

```bash
python .uploads/experiments/g3/run_g3_p1_cloud.py \
  --stage prepare \
  --run-dir "$RUN_DIR" \
  --request-prefixes "$PREFIXES" \
  --trace "$TRACE"
```

该阶段直接从 `$PREFIXES` 生成：

```text
$TRACE
$TRACE.manifest.json
```

不会生成体积很大的中间 `access_trace.jsonl`。若 trace 已存在且 manifest
一致，则复用；默认拒绝覆盖。确需重建时显式增加
`--force-rebuild-trace`。

### 阶段 B：只在 validation 上扫描 54 组参数

```bash
python .uploads/experiments/g3/run_g3_p1_cloud.py \
  --stage tune \
  --run-dir "$RUN_DIR" \
  --request-prefixes "$PREFIXES" \
  --trace "$TRACE"
```

查看选择结果：

```bash
python -m json.tool "$RUN_DIR/tuning/selection.json" | less
```

只有 `status = SELECTED` 时才会生成：

```text
$RUN_DIR/frozen-config.yaml
```

调参器只解析、因果标注一次完整 trace；固定 baseline 与相同 CPU 参数结果
也会复用。54 组候选对应 76 次 baseline replay，而不是 324 次。

若结果是 `NO_VALID_CONFIG`，到此停止，不得运行 held-out test。
`NO_VALIDATION_DATA` 表示 `--max-episodes` 太小，导致 task_id 分组后验证集为空；
应增大上限或使用完整 trace，而不是改 split 后偷看 test。

### 阶段 C：冻结参数后只运行一次 held-out test

```bash
python .uploads/experiments/g3/run_g3_p1_cloud.py \
  --stage test \
  --run-dir "$RUN_DIR" \
  --request-prefixes "$PREFIXES" \
  --trace "$TRACE"
```

同一 `RUN_DIR` 已存在 held-out 结果时，脚本会拒绝第二次运行。主要产物：

```text
$RUN_DIR/g3-p1-held-out-test.csv
$RUN_DIR/g3-p1-check.json
$RUN_DIR/logs/held-out-test.log
$RUN_DIR/logs/protocol-check.log
```

检查：

```bash
python -m json.tool "$RUN_DIR/g3-p1-check.json" | less
```

`PASS` 仅表示 G3-P1 单 cell 工程协议通过；下一步仍是 closed-loop serving
测量真实 TTFT、achieved throughput 和 SLO goodput。

## 4. 一条命令执行

只建议在新的 `RUN_DIR` 使用：

```bash
python .uploads/experiments/g3/run_g3_p1_cloud.py \
  --stage all \
  --run-dir "$RUN_DIR" \
  --request-prefixes "$PREFIXES" \
  --trace "$TRACE"
```

如果 validation 没有有效配置，流程会在冻结阶段停止，不会读取 test。

## 5. 小规模云端 smoke

用于确认环境，不可作为实验结果：

```bash
export SMOKE_DIR="/root/autodl-tmp/flowcache-runs/g3-smoke-$(date -u +%Y%m%dT%H%M%SZ)"

python .uploads/experiments/g1prime/build_concurrent_access_trace.py \
  --input "$PREFIXES" \
  --output "$SMOKE_DIR/access_trace_c4.jsonl" \
  --manifest "$SMOKE_DIR/access_trace_c4.jsonl.manifest.json" \
  --concurrency 4 \
  --max-requests 100
```

正式流程不得把该 100-request trace 传给 validation/test。

## 6. 资源不足

若预检报告的保守内存估计高于 `MemAvailable`，优先换大内存实例。确认已用
小规模 trace 监控过峰值后，才可显式覆盖：

```bash
python .uploads/experiments/g3/run_g3_p1_cloud.py \
  --stage prepare \
  --run-dir "$RUN_DIR" \
  --request-prefixes "$PREFIXES" \
  --trace "$TRACE" \
  --allow-low-memory
```

不要用 swap 抖动下的 replay wall 形成控制器开销结论。
