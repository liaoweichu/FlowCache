# Checklist

## Track 1: 14 周执行计划
- [x] 14 周计划文档存在且覆盖 W1-W14 每周目标
- [x] 每周映射到 ccfa.yaml 中的 gate (G0-G5) 或 experiment (E1-E7)
- [x] 每个 gate 的失败动作和回退路线已定义
- [x] 关键路径已标注（G0→G1→G3→G4→G5 的依赖链）
- [x] 路线 A / B / C 的切换触发条件已整合
- [x] stage gate 状态追踪机制可用（ccfa.yaml 字段或追踪文件）

## Track 2: Prior Art 核验
- [x] 6 篇核心论文均已核验存在性（存在 / 不存在 / 无法确认）
- [x] 每篇论文的内容摘要已记录（是否含 workflow next-use、fidelity-aware precision、联合 residency）
- [x] PBKV 是否已含 fidelity-aware precision 已明确判定
- [x] QKVShare 是否已含长生命周期驻留已明确判定
- [x] GraphFlow 的 ICML 2026 venue 声明已核验
- [x] novelty delta 判定已给出（保留 / 崩塌 / 部分崩塌）
- [x] references.bib 的 unverified 标注已根据核验结果更新
- [x] 核验报告文件已产出

## Track 3: G2 Pilot 实验设计
- [x] pilot 数据集已选定（含子集大小和选择理由）
- [x] R 标签采集协议完整（next-use 定义、测量方法、saved-prefill 计量）
- [x] D 标签采集协议完整（量化精度选择、质量指标、干预回放方法）
- [x] 统计检验方法已选定（相关系数类型、显著性水平、功效分析）
- [x] go / no-go 判定阈值已定义（含中间灰区处理）
- [x] 实验设计文档已写入 experiments/ 目录
- [x] 设计文档可在 50-100 workflow 上实际执行
- [x] 数据集使用真实数据子集（非伪造数据）
- [x] 硬件约束（RTX 4090D 24GB）已在设计中考虑
