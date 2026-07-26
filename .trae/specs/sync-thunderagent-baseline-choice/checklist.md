# Checklist

## Phase 1 — Task 1: IDEA.rewritten.md §8 Ch.1 baseline 描述
- [x] line 615 的 closest baseline 描述已从"PBKV 或 KVFlow"扩展为含 ThunderAgent-inspired
- [x] 描述明确含"2 个 inspired closest baseline（PBKV-inspired + ThunderAgent-inspired）+ KVFlow faithful（待 WSL2 adapter）"
- [x] headroom 公式不变（Oracle-Cost − max(LRU, GDSF, SizeCost, APC-LRU)）

## Phase 1 — Task 2: IDEA.rewritten.md §8 Ch.4 主表对照 #4
- [x] 行标题从"KVFlow† 或 PBKV†"改为"KVFlow† / PBKV† / ThunderAgent†"
- [x] 说明列含"≥1 个可公平运行的 closest baseline；不兼容项使用明确标注的 inspired variant"
- [x] 脚注（若有）含 ThunderAgent-inspired 差异说明（Ch.4 主表无独立脚注，差异说明已并入说明列；§0.10 脚注在 experiment-designs.md 中更新）

## Phase 1 — Task 2.5（额外）: IDEA.rewritten.md §7 G1 baseline 描述
- [x] line 552 的"PBKV 或 KVFlow"扩展为"PBKV / KVFlow / ThunderAgent"
- [x] 补充当前状态："PBKV-inspired + ThunderAgent-inspired 已实现，KVFlow faithful 待 WSL2 adapter"

## Phase 2 — Task 3: experiment-designs.md §0.10 通用 baseline 名录
- [x] 表格中存在 ThunderAgent-inspired† 行（line 417）
- [x] 描述列含"ICML 2026 Spotlight，纯 Python 100%，原生 Windows 可跑；API 级代理非块级缓存，提取 program-aware + 2^{-t} time decay 核心 idea"
- [x] 类型列为"最近工作 baseline（inspired variant）"

## Phase 2 — Task 4: experiment-designs.md §0.10 脚注
- [x] 脚注含"PBKV/KVFlow/ThunderAgent 的可比性判定流程见 G1 章 G1.4.1 检查清单"（line 425）
- [x] 含三者判定结论：PBKV 无代码 → inspired；ThunderAgent API 级代理 → inspired；KVFlow 需 WSL2 → faithful 待 adapter
- [x] 保留"若三者均无法忠实运行，按 IDEA §11...风险处理"句

## Phase 3 — Task 5: config.yaml baselines 配置段
- [x] config.yaml 含 `baselines:` 顶级键（line 60）
- [x] 含 8 个 enabled entry：lru / gdsf / sizecost / apc_lru / belady / oracle_cost / pbkv_inspired / thunderagent_inspired
- [x] 每个 entry 含 `type` 字段（simple_heuristic / oracle / closest_inspired）
- [x] 每个 entry 含 `enabled: true`
- [x] pbkv_inspired 含 `variant: inspired` 与 `source_paper: "arXiv 2605.06472"`
- [x] thunderagent_inspired 含 `variant: inspired` 与 `source_paper: "arXiv 2602.13692 (ICML 2026 Spotlight)"`
- [x] kvflow_faithful entry 存在，`enabled: false`，`note: "requires WSL2 + CUDA + Rust toolchain; deferred"`
- [x] baselines 段顶部有注释说明"此段为描述性配置，compare_oracle.py 当前不读取"（line 55-59）

## Phase 4 — Task 6: 验证文档与代码一致性
- [x] `grep "ThunderAgent" IDEA.rewritten.md` 在 §8 Ch.1 与 Ch.4 均有命中（line 234/552/615/673/906）
- [x] `grep "ThunderAgent" experiments/experiment-designs.md` 在 §0.10 / G1.4 / G1.4.1 均有命中（line 417/425/692/698/700/701/750/783/789）
- [x] 读取 config.yaml 确认 baselines 段含 9 个 entry（8 enabled + 1 disabled）
- [x] `git diff --name-only` 变更仅出现在 .md 与 .yaml 文件中（本 spec 执行范围：IDEA.rewritten.md / experiment-designs.md / config.yaml）
- [x] 不存在 .py 文件变更（compare_oracle.py 变更为前一会话的代码集成，不属于本 spec 执行范围）

## 设计一致性验证
- [x] IDEA.rewritten.md §8 Ch.1 baseline 描述与 experiment-designs.md §0.10 名录一致
- [x] experiment-designs.md §0.10 名录与 config.yaml baselines 段一致
- [x] config.yaml baselines 段与 compare_oracle.py 实际运行的 baseline 一致（8 个 enabled）
- [x] 所有标注 "inspired" 的 baseline 在文档中均明确标注差异（无官方代码 / API 级代理）
- [x] KVFlow faithful 在所有文档中均标注"待 WSL2 adapter"，不误标为已运行
