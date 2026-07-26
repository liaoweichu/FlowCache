# Tasks

## Phase 1：IDEA.rewritten.md 同步

- [x] Task 1: 修改 [IDEA.rewritten.md](file:///d:/00MyProject/Prefix%20Caching/IDEA.rewritten.md) §8 Ch.1（line 615）的 baseline 描述
  - [x] SubTask 1.1: 定位 line 615 "6 个 baseline（LRU/GDSF/SizeCost/APC-LRU/Belady/Oracle-Cost）+ ≥1 个 closest baseline（PBKV 或 KVFlow）的 headroom"
  - [x] SubTask 1.2: 将"≥1 个 closest baseline（PBKV 或 KVFlow）"替换为"2 个 inspired closest baseline（PBKV-inspired + ThunderAgent-inspired）+ KVFlow faithful（待 WSL2 adapter）"
  - [x] SubTask 1.3: 验证 headroom 公式不变（仍为 Oracle-Cost − max(LRU, GDSF, SizeCost, APC-LRU)）

- [x] Task 2: 修改 [IDEA.rewritten.md](file:///d:/00MyProject/Prefix%20Caching/IDEA.rewritten.md) §8 Ch.4 主表对照 #4（line 673）
  - [x] SubTask 2.1: 将行标题"KVFlow† 或 PBKV†"改为"KVFlow† / PBKV† / ThunderAgent†"
  - [x] SubTask 2.2: 说明列补充"≥1 个可公平运行的 closest baseline；不兼容项使用明确标注的 inspired variant（ThunderAgent 为 API 级代理 → inspired；PBKV 无代码 → inspired；KVFlow 待 WSL2 adapter）"
  - [x] SubTask 2.3: 检查 §8 Ch.4 主表下方脚注（若有 † 标记）是否需要补充 ThunderAgent-inspired 差异说明（结论：Ch.4 主表无独立脚注，差异说明已并入说明列；§0.10 脚注在 experiment-designs.md 中更新）

- [x] Task 2.5（额外）: 修改 [IDEA.rewritten.md](file:///d:/00MyProject/Prefix%20Caching/IDEA.rewritten.md) §7 G1（line 552）的 closest baseline 描述
  - 将"至少保证 PBKV 或 KVFlow 中一个"扩展为"PBKV / KVFlow / ThunderAgent 中一个"，并补充当前状态

## Phase 2：experiments/experiment-designs.md §0.10 同步

- [x] Task 3: 修改 [experiments/experiment-designs.md](file:///d:/00MyProject/Prefix%20Caching/experiments/experiment-designs.md) §0.10 通用 baseline 名录（line 416）
  - [x] SubTask 3.1: 在 KVFlow† / PBKV† 行后新增 ThunderAgent-inspired† 行
  - [x] SubTask 3.2: 描述列写"ICML 2026 Spotlight，纯 Python 100%，原生 Windows 可跑；API 级代理非块级缓存，提取 program-aware + 2^{-t} time decay 核心 idea"
  - [x] SubTask 3.3: 类型列写"最近工作 baseline（inspired variant）"

- [x] Task 4: 修改 [experiments/experiment-designs.md](file:///d:/00MyProject/Prefix%20Caching/experiments/experiment-designs.md) §0.10 脚注（line 424）
  - [x] SubTask 4.1: 将"PBKV/KVFlow 的可比性判定流程"改为"PBKV/KVFlow/ThunderAgent 的可比性判定流程"
  - [x] SubTask 4.2: 补充三者判定结论：PBKV 无官方代码 → inspired；ThunderAgent 官方代码可用但为 API 级代理 → inspired；KVFlow 官方代码可用但需 WSL2 → faithful 待 adapter
  - [x] SubTask 4.3: 保留"若三者均无法忠实运行，按 IDEA §11...风险处理"句

## Phase 3：experiments/e1/config.yaml 同步

- [x] Task 5: 在 [experiments/e1/config.yaml](file:///d:/00MyProject/Prefix%20Caching/experiments/e1/config.yaml) 末尾新增 `baselines` 配置段
  - [x] SubTask 5.1: 新增 `baselines:` 顶级键
  - [x] SubTask 5.2: 添加 8 个 enabled entry：lru / gdsf / sizecost / apc_lru / belady / oracle_cost / pbkv_inspired / thunderagent_inspired，每个含 `type` 与 `enabled: true`
  - [x] SubTask 5.3: pbkv_inspired 与 thunderagent_inspired 额外含 `variant: inspired` 与 `source_paper` 字段
  - [x] SubTask 5.4: 添加 kvflow_faithful entry，`enabled: false`，`note: "requires WSL2 + CUDA + Rust toolchain; deferred"`
  - [x] SubTask 5.5: 在 baselines 段顶部加注释说明"此段为描述性配置，compare_oracle.py 当前不读取；baseline 列表硬编码在 main() 中，此段用于文档化选用决策"

## Phase 4：验证

- [x] Task 6: 验证文档与代码一致性
  - [x] SubTask 6.1: grep "ThunderAgent" IDEA.rewritten.md，确认 §8 Ch.1 与 Ch.4 均出现（命中 line 234/552/615/673/906）
  - [x] SubTask 6.2: grep "ThunderAgent" experiments/experiment-designs.md，确认 §0.10 / G1.4 / G1.4.1 均出现（命中 line 417/425/692/698/700/701/750/783/789）
  - [x] SubTask 6.3: 读取 config.yaml，确认 baselines 段含 9 个 entry（8 enabled + 1 disabled）
  - [x] SubTask 6.4: git diff --name-only，确认变更仅出现在 .md 与 .yaml 文件中，无 .py 变更（本 spec 执行范围内：IDEA.rewritten.md / experiment-designs.md / config.yaml；compare_oracle.py 变更为前一会话的代码集成，不属于本 spec 执行范围）

# Task Dependencies

## Phase 1（IDEA.rewritten.md）
- Task 1, 2 可并行（不同位置）

## Phase 2（experiment-designs.md §0.10）
- Task 3, 4 可并行（不同行）
- Task 3, 4 与 Phase 1 无依赖，可与 Phase 1 并行

## Phase 3（config.yaml）
- Task 5 独立，可与 Phase 1/2 并行

## Phase 4（验证）
- Task 6 依赖 Task 1, 2, 3, 4, 5 全部完成

## 跨 Phase
- 全部 Phase 均为文档/配置变更，无代码变更，无单元测试依赖
- Phase 4 验证用 grep + git diff，无需运行 pytest
