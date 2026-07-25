# G0 Exactness Report

BF16 缓存恢复 vs 重算一致性测试 + block identity 正确性验证。

## 表 G0-1: BF16 缓存恢复 vs 重算数值一致性

| Case ID | Category | Seq Len | Num Blocks | KV Bit-Identical | Logits Max Abs Diff | Logits Mean Abs Diff | Cosine Sim | Top-1 Match |
|---------|----------|---------|------------|------------------|---------------------|----------------------|------------|-------------|
| cat1_retail_000_p0 | 1 | 179 | 12 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat1_retail_000_p1 | 1 | 177 | 12 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat1_retail_001_p0 | 1 | 180 | 12 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat1_retail_001_p1 | 1 | 170 | 11 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat1_retail_002_p0 | 1 | 179 | 12 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat1_retail_002_p1 | 1 | 179 | 12 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat1_retail_003_p0 | 1 | 177 | 12 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat1_retail_003_p1 | 1 | 179 | 12 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat1_retail_004_p0 | 1 | 180 | 12 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat1_retail_004_p1 | 1 | 180 | 12 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat1_retail_005_p0 | 1 | 179 | 12 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat1_retail_005_p1 | 1 | 174 | 11 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat1_retail_006_p0 | 1 | 177 | 12 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat1_retail_006_p1 | 1 | 175 | 11 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat1_retail_007_p0 | 1 | 177 | 12 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat1_retail_007_p1 | 1 | 177 | 12 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat1_retail_008_p0 | 1 | 174 | 11 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat1_retail_008_p1 | 1 | 175 | 11 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat1_retail_009_p0 | 1 | 177 | 12 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat1_retail_009_p1 | 1 | 177 | 12 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat1_retail_010_p0 | 1 | 177 | 12 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat1_retail_010_p1 | 1 | 175 | 11 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat1_retail_011_p0 | 1 | 174 | 11 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat1_retail_011_p1 | 1 | 173 | 11 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat1_retail_012_p0 | 1 | 175 | 11 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat1_retail_012_p1 | 1 | 170 | 11 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat1_retail_013_p0 | 1 | 172 | 11 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat1_retail_013_p1 | 1 | 172 | 11 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat1_retail_014_p0 | 1 | 172 | 11 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat1_retail_014_p1 | 1 | 176 | 11 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat1_airline_000_p0 | 1 | 149 | 10 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat1_airline_000_p1 | 1 | 152 | 10 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat1_airline_001_p0 | 1 | 150 | 10 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat1_airline_001_p1 | 1 | 150 | 10 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat1_airline_002_p0 | 1 | 145 | 10 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat1_airline_002_p1 | 1 | 149 | 10 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat1_airline_003_p0 | 1 | 146 | 10 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat1_airline_003_p1 | 1 | 152 | 10 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat1_airline_004_p0 | 1 | 147 | 10 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat1_airline_004_p1 | 1 | 148 | 10 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat1_airline_005_p0 | 1 | 152 | 10 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat1_airline_005_p1 | 1 | 148 | 10 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat1_airline_006_p0 | 1 | 143 | 9 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat1_airline_006_p1 | 1 | 147 | 10 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat1_airline_007_p0 | 1 | 145 | 10 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat1_airline_007_p1 | 1 | 151 | 10 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat1_airline_008_p0 | 1 | 150 | 10 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat1_airline_008_p1 | 1 | 147 | 10 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat1_airline_009_p0 | 1 | 144 | 9 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat1_airline_009_p1 | 1 | 146 | 10 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat1_airline_010_p0 | 1 | 145 | 10 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat1_airline_010_p1 | 1 | 146 | 10 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat1_airline_011_p0 | 1 | 147 | 10 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat1_airline_011_p1 | 1 | 148 | 10 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat1_airline_012_p0 | 1 | 149 | 10 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat1_airline_012_p1 | 1 | 149 | 10 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat1_airline_013_p0 | 1 | 149 | 10 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat1_airline_013_p1 | 1 | 144 | 9 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat1_airline_014_p0 | 1 | 146 | 10 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat1_airline_014_p1 | 1 | 151 | 10 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat2_retail_single_000_p0 | 2 | 209 | 14 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat2_retail_single_000_p1 | 2 | 211 | 14 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat2_retail_single_001_p0 | 2 | 207 | 13 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat2_retail_single_001_p1 | 2 | 209 | 14 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat2_retail_single_002_p0 | 2 | 210 | 14 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat2_retail_single_002_p1 | 2 | 212 | 14 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat2_retail_single_003_p0 | 2 | 200 | 13 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat2_retail_single_003_p1 | 2 | 202 | 13 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat2_retail_single_004_p0 | 2 | 209 | 14 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat2_retail_single_004_p1 | 2 | 211 | 14 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat2_retail_single_005_p0 | 2 | 209 | 14 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat2_retail_single_005_p1 | 2 | 211 | 14 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat2_retail_single_006_p0 | 2 | 207 | 13 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat2_retail_single_006_p1 | 2 | 209 | 14 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat2_retail_single_007_p0 | 2 | 209 | 14 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat2_retail_single_007_p1 | 2 | 211 | 14 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat2_retail_single_008_p0 | 2 | 210 | 14 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat2_retail_single_008_p1 | 2 | 212 | 14 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat2_retail_single_009_p0 | 2 | 210 | 14 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat2_retail_single_009_p1 | 2 | 212 | 14 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat2_airline_single_010_p0 | 2 | 182 | 12 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat2_airline_single_010_p1 | 2 | 184 | 12 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat2_airline_single_011_p0 | 2 | 178 | 12 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat2_airline_single_011_p1 | 2 | 180 | 12 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat2_airline_single_012_p0 | 2 | 173 | 11 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat2_airline_single_012_p1 | 2 | 175 | 11 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat2_airline_single_013_p0 | 2 | 177 | 12 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat2_airline_single_013_p1 | 2 | 179 | 12 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat2_airline_single_014_p0 | 2 | 175 | 11 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat2_airline_single_014_p1 | 2 | 177 | 12 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat2_airline_single_015_p0 | 2 | 181 | 12 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat2_airline_single_015_p1 | 2 | 183 | 12 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat2_airline_single_016_p0 | 2 | 180 | 12 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat2_airline_single_016_p1 | 2 | 182 | 12 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat2_airline_single_017_p0 | 2 | 177 | 12 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat2_airline_single_017_p1 | 2 | 179 | 12 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat2_airline_single_018_p0 | 2 | 174 | 11 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat2_airline_single_018_p1 | 2 | 176 | 11 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat2_airline_single_019_p0 | 2 | 176 | 11 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat2_airline_single_019_p1 | 2 | 178 | 12 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat2_retail_multi_000_p0 | 2 | 241 | 16 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat2_retail_multi_000_p1 | 2 | 242 | 16 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat2_retail_multi_001_p0 | 2 | 239 | 15 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat2_retail_multi_001_p1 | 2 | 240 | 15 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat2_retail_multi_002_p0 | 2 | 242 | 16 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat2_retail_multi_002_p1 | 2 | 243 | 16 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat2_retail_multi_003_p0 | 2 | 232 | 15 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat2_retail_multi_003_p1 | 2 | 233 | 15 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat2_retail_multi_004_p0 | 2 | 241 | 16 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat2_retail_multi_004_p1 | 2 | 242 | 16 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat2_airline_multi_005_p0 | 2 | 211 | 14 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat2_airline_multi_005_p1 | 2 | 212 | 14 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat2_airline_multi_006_p0 | 2 | 208 | 13 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat2_airline_multi_006_p1 | 2 | 209 | 14 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat2_airline_multi_007_p0 | 2 | 214 | 14 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat2_airline_multi_007_p1 | 2 | 215 | 14 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat2_airline_multi_008_p0 | 2 | 209 | 14 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat2_airline_multi_008_p1 | 2 | 210 | 14 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat2_airline_multi_009_p0 | 2 | 210 | 14 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat2_airline_multi_009_p1 | 2 | 211 | 14 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat3_000_p0 | 3 | 179 | 12 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat3_000_p1 | 3 | 179 | 12 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat3_001_p0 | 3 | 177 | 12 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat3_001_p1 | 3 | 177 | 12 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat3_002_p0 | 3 | 180 | 12 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat3_002_p1 | 3 | 180 | 12 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat3_003_p0 | 3 | 170 | 11 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat3_003_p1 | 3 | 170 | 11 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat3_004_p0 | 3 | 179 | 12 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat3_004_p1 | 3 | 179 | 12 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat3_005_p0 | 3 | 149 | 10 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat3_005_p1 | 3 | 149 | 10 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat3_006_p0 | 3 | 146 | 10 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat3_006_p1 | 3 | 146 | 10 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat3_007_p0 | 3 | 152 | 10 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat3_007_p1 | 3 | 152 | 10 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat3_008_p0 | 3 | 147 | 10 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat3_008_p1 | 3 | 147 | 10 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat3_009_p0 | 3 | 148 | 10 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat3_009_p1 | 3 | 148 | 10 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat4_000_p0 | 4 | 179 | 12 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat4_000_p1 | 4 | 179 | 12 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat4_001_p0 | 4 | 177 | 12 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat4_001_p1 | 4 | 177 | 12 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat4_002_p0 | 4 | 180 | 12 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat4_002_p1 | 4 | 180 | 12 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat4_003_p0 | 4 | 170 | 11 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat4_003_p1 | 4 | 170 | 11 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat4_004_p0 | 4 | 179 | 12 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat4_004_p1 | 4 | 179 | 12 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat4_005_p0 | 4 | 149 | 10 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat4_005_p1 | 4 | 149 | 10 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat4_006_p0 | 4 | 146 | 10 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat4_006_p1 | 4 | 146 | 10 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat4_007_p0 | 4 | 152 | 10 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat4_007_p1 | 4 | 152 | 10 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat4_008_p0 | 4 | 147 | 10 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat4_008_p1 | 4 | 147 | 10 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat4_009_p0 | 4 | 148 | 10 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat4_009_p1 | 4 | 148 | 10 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat5_000_t0 | 5 | 174 | 11 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat5_000_t1 | 5 | 204 | 13 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat5_000_t2 | 5 | 241 | 16 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat5_000_t3 | 5 | 273 | 18 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat5_001_t0 | 5 | 180 | 12 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat5_001_t1 | 5 | 214 | 14 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat5_001_t2 | 5 | 239 | 15 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat5_001_t3 | 5 | 273 | 18 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat5_002_t0 | 5 | 179 | 12 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat5_002_t1 | 5 | 207 | 13 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat5_002_t2 | 5 | 241 | 16 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat5_002_t3 | 5 | 285 | 18 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat5_003_t0 | 5 | 179 | 12 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat5_003_t1 | 5 | 215 | 14 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat5_003_t2 | 5 | 255 | 16 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat5_003_t3 | 5 | 282 | 18 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat5_004_t0 | 5 | 174 | 11 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat5_004_t1 | 5 | 207 | 13 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat5_004_t2 | 5 | 294 | 19 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat5_004_t3 | 5 | 331 | 21 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat5_005_t0 | 5 | 149 | 10 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat5_005_t1 | 5 | 183 | 12 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat5_005_t2 | 5 | 212 | 14 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat5_005_t3 | 5 | 247 | 16 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat5_006_t0 | 5 | 150 | 10 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat5_006_t1 | 5 | 187 | 12 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat5_006_t2 | 5 | 217 | 14 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat5_006_t3 | 5 | 254 | 16 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat5_007_t0 | 5 | 149 | 10 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat5_007_t1 | 5 | 180 | 12 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat5_007_t2 | 5 | 209 | 14 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat5_007_t3 | 5 | 248 | 16 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat5_008_t0 | 5 | 150 | 10 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat5_008_t1 | 5 | 196 | 13 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat5_008_t2 | 5 | 229 | 15 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat5_008_t3 | 5 | 265 | 17 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat5_009_t0 | 5 | 148 | 10 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat5_009_t1 | 5 | 189 | 12 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat5_009_t2 | 5 | 222 | 14 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat5_009_t3 | 5 | 250 | 16 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat6_000_p0 | 6 | 179 | 12 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat6_000_p1 | 6 | 149 | 10 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat6_001_p0 | 6 | 177 | 12 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat6_001_p1 | 6 | 152 | 10 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat6_002_p0 | 6 | 180 | 12 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat6_002_p1 | 6 | 150 | 10 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat6_003_p0 | 6 | 170 | 11 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat6_003_p1 | 6 | 150 | 10 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat6_004_p0 | 6 | 179 | 12 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat6_004_p1 | 6 | 145 | 10 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat6_005_p0 | 6 | 179 | 12 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat6_005_p1 | 6 | 149 | 10 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat6_006_p0 | 6 | 177 | 12 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat6_006_p1 | 6 | 146 | 10 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat6_007_p0 | 6 | 179 | 12 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat6_007_p1 | 6 | 152 | 10 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat6_008_p0 | 6 | 180 | 12 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat6_008_p1 | 6 | 147 | 10 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat6_009_p0 | 6 | 180 | 12 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |
| cat6_009_p1 | 6 | 148 | 10 | ✓ | 0.00e+00 | 0.00e+00 | 1.000000 | ✓ |

**汇总统计：**

- 测试序列总数: 220
- KV bit-identical: 220/220
- Logits max abs diff ≤ 1e-3: 220/220
- Top-1 token 一致: 220/220

## 表 G0-2: Block Identity / 父链 / Invalidation 正确性

| Case ID | Category | Identity Check | Parent Chain | Invalidation | Detail |
|---------|----------|----------------|--------------|--------------|--------|
| cat1_retail_000 | 1 | ✓ PASS | ✓ PASS | N/A | common_prefix=9, blocks_a=12, blocks_b=12, rest_differ=True |
| cat1_retail_001 | 1 | ✓ PASS | ✓ PASS | N/A | common_prefix=9, blocks_a=12, blocks_b=11, rest_differ=True |
| cat1_retail_002 | 1 | ✓ PASS | ✓ PASS | N/A | common_prefix=9, blocks_a=12, blocks_b=12, rest_differ=True |
| cat1_retail_003 | 1 | ✓ PASS | ✓ PASS | N/A | common_prefix=9, blocks_a=12, blocks_b=12, rest_differ=True |
| cat1_retail_004 | 1 | ✓ PASS | ✓ PASS | N/A | common_prefix=9, blocks_a=12, blocks_b=12, rest_differ=True |
| cat1_retail_005 | 1 | ✓ PASS | ✓ PASS | N/A | common_prefix=9, blocks_a=12, blocks_b=11, rest_differ=True |
| cat1_retail_006 | 1 | ✓ PASS | ✓ PASS | N/A | common_prefix=9, blocks_a=12, blocks_b=11, rest_differ=True |
| cat1_retail_007 | 1 | ✓ PASS | ✓ PASS | N/A | common_prefix=9, blocks_a=12, blocks_b=12, rest_differ=True |
| cat1_retail_008 | 1 | ✓ PASS | ✓ PASS | N/A | common_prefix=9, blocks_a=11, blocks_b=11, rest_differ=True |
| cat1_retail_009 | 1 | ✓ PASS | ✓ PASS | N/A | common_prefix=9, blocks_a=12, blocks_b=12, rest_differ=True |
| cat1_retail_010 | 1 | ✓ PASS | ✓ PASS | N/A | common_prefix=9, blocks_a=12, blocks_b=11, rest_differ=True |
| cat1_retail_011 | 1 | ✓ PASS | ✓ PASS | N/A | common_prefix=9, blocks_a=11, blocks_b=11, rest_differ=True |
| cat1_retail_012 | 1 | ✓ PASS | ✓ PASS | N/A | common_prefix=9, blocks_a=11, blocks_b=11, rest_differ=True |
| cat1_retail_013 | 1 | ✓ PASS | ✓ PASS | N/A | common_prefix=9, blocks_a=11, blocks_b=11, rest_differ=True |
| cat1_retail_014 | 1 | ✓ PASS | ✓ PASS | N/A | common_prefix=9, blocks_a=11, blocks_b=11, rest_differ=True |
| cat1_airline_000 | 1 | ✓ PASS | ✓ PASS | N/A | common_prefix=8, blocks_a=10, blocks_b=10, rest_differ=True |
| cat1_airline_001 | 1 | ✓ PASS | ✓ PASS | N/A | common_prefix=8, blocks_a=10, blocks_b=10, rest_differ=True |
| cat1_airline_002 | 1 | ✓ PASS | ✓ PASS | N/A | common_prefix=8, blocks_a=10, blocks_b=10, rest_differ=True |
| cat1_airline_003 | 1 | ✓ PASS | ✓ PASS | N/A | common_prefix=8, blocks_a=10, blocks_b=10, rest_differ=True |
| cat1_airline_004 | 1 | ✓ PASS | ✓ PASS | N/A | common_prefix=8, blocks_a=10, blocks_b=10, rest_differ=True |
| cat1_airline_005 | 1 | ✓ PASS | ✓ PASS | N/A | common_prefix=8, blocks_a=10, blocks_b=10, rest_differ=True |
| cat1_airline_006 | 1 | ✓ PASS | ✓ PASS | N/A | common_prefix=8, blocks_a=9, blocks_b=10, rest_differ=True |
| cat1_airline_007 | 1 | ✓ PASS | ✓ PASS | N/A | common_prefix=8, blocks_a=10, blocks_b=10, rest_differ=True |
| cat1_airline_008 | 1 | ✓ PASS | ✓ PASS | N/A | common_prefix=8, blocks_a=10, blocks_b=10, rest_differ=True |
| cat1_airline_009 | 1 | ✓ PASS | ✓ PASS | N/A | common_prefix=8, blocks_a=9, blocks_b=10, rest_differ=True |
| cat1_airline_010 | 1 | ✓ PASS | ✓ PASS | N/A | common_prefix=8, blocks_a=10, blocks_b=10, rest_differ=True |
| cat1_airline_011 | 1 | ✓ PASS | ✓ PASS | N/A | common_prefix=8, blocks_a=10, blocks_b=10, rest_differ=True |
| cat1_airline_012 | 1 | ✓ PASS | ✓ PASS | N/A | common_prefix=8, blocks_a=10, blocks_b=10, rest_differ=True |
| cat1_airline_013 | 1 | ✓ PASS | ✓ PASS | N/A | common_prefix=8, blocks_a=10, blocks_b=9, rest_differ=True |
| cat1_airline_014 | 1 | ✓ PASS | ✓ PASS | N/A | common_prefix=8, blocks_a=10, blocks_b=10, rest_differ=True |
| cat2_retail_single_000 | 2 | ✓ PASS | ✓ PASS | N/A | common_prefix=12, blocks_a=14, blocks_b=14, rest_differ=True |
| cat2_retail_single_001 | 2 | ✓ PASS | ✓ PASS | N/A | common_prefix=12, blocks_a=13, blocks_b=14, rest_differ=True |
| cat2_retail_single_002 | 2 | ✓ PASS | ✓ PASS | N/A | common_prefix=12, blocks_a=14, blocks_b=14, rest_differ=True |
| cat2_retail_single_003 | 2 | ✓ PASS | ✓ PASS | N/A | common_prefix=11, blocks_a=13, blocks_b=13, rest_differ=True |
| cat2_retail_single_004 | 2 | ✓ PASS | ✓ PASS | N/A | common_prefix=12, blocks_a=14, blocks_b=14, rest_differ=True |
| cat2_retail_single_005 | 2 | ✓ PASS | ✓ PASS | N/A | common_prefix=12, blocks_a=14, blocks_b=14, rest_differ=True |
| cat2_retail_single_006 | 2 | ✓ PASS | ✓ PASS | N/A | common_prefix=12, blocks_a=13, blocks_b=14, rest_differ=True |
| cat2_retail_single_007 | 2 | ✓ PASS | ✓ PASS | N/A | common_prefix=12, blocks_a=14, blocks_b=14, rest_differ=True |
| cat2_retail_single_008 | 2 | ✓ PASS | ✓ PASS | N/A | common_prefix=12, blocks_a=14, blocks_b=14, rest_differ=True |
| cat2_retail_single_009 | 2 | ✓ PASS | ✓ PASS | N/A | common_prefix=12, blocks_a=14, blocks_b=14, rest_differ=True |
| cat2_airline_single_010 | 2 | ✓ PASS | ✓ PASS | N/A | common_prefix=10, blocks_a=12, blocks_b=12, rest_differ=True |
| cat2_airline_single_011 | 2 | ✓ PASS | ✓ PASS | N/A | common_prefix=10, blocks_a=12, blocks_b=12, rest_differ=True |
| cat2_airline_single_012 | 2 | ✓ PASS | ✓ PASS | N/A | common_prefix=10, blocks_a=11, blocks_b=11, rest_differ=True |
| cat2_airline_single_013 | 2 | ✓ PASS | ✓ PASS | N/A | common_prefix=10, blocks_a=12, blocks_b=12, rest_differ=True |
| cat2_airline_single_014 | 2 | ✓ PASS | ✓ PASS | N/A | common_prefix=10, blocks_a=11, blocks_b=12, rest_differ=True |
| cat2_airline_single_015 | 2 | ✓ PASS | ✓ PASS | N/A | common_prefix=10, blocks_a=12, blocks_b=12, rest_differ=True |
| cat2_airline_single_016 | 2 | ✓ PASS | ✓ PASS | N/A | common_prefix=10, blocks_a=12, blocks_b=12, rest_differ=True |
| cat2_airline_single_017 | 2 | ✓ PASS | ✓ PASS | N/A | common_prefix=10, blocks_a=12, blocks_b=12, rest_differ=True |
| cat2_airline_single_018 | 2 | ✓ PASS | ✓ PASS | N/A | common_prefix=10, blocks_a=11, blocks_b=11, rest_differ=True |
| cat2_airline_single_019 | 2 | ✓ PASS | ✓ PASS | N/A | common_prefix=10, blocks_a=11, blocks_b=12, rest_differ=True |
| cat2_retail_multi_000 | 2 | ✓ PASS | ✓ PASS | N/A | common_prefix=14, blocks_a=16, blocks_b=16, rest_differ=True |
| cat2_retail_multi_001 | 2 | ✓ PASS | ✓ PASS | N/A | common_prefix=14, blocks_a=15, blocks_b=15, rest_differ=True |
| cat2_retail_multi_002 | 2 | ✓ PASS | ✓ PASS | N/A | common_prefix=14, blocks_a=16, blocks_b=16, rest_differ=True |
| cat2_retail_multi_003 | 2 | ✓ PASS | ✓ PASS | N/A | common_prefix=13, blocks_a=15, blocks_b=15, rest_differ=True |
| cat2_retail_multi_004 | 2 | ✓ PASS | ✓ PASS | N/A | common_prefix=14, blocks_a=16, blocks_b=16, rest_differ=True |
| cat2_airline_multi_005 | 2 | ✓ PASS | ✓ PASS | N/A | common_prefix=12, blocks_a=14, blocks_b=14, rest_differ=True |
| cat2_airline_multi_006 | 2 | ✓ PASS | ✓ PASS | N/A | common_prefix=12, blocks_a=13, blocks_b=14, rest_differ=True |
| cat2_airline_multi_007 | 2 | ✓ PASS | ✓ PASS | N/A | common_prefix=12, blocks_a=14, blocks_b=14, rest_differ=True |
| cat2_airline_multi_008 | 2 | ✓ PASS | ✓ PASS | N/A | common_prefix=12, blocks_a=14, blocks_b=14, rest_differ=True |
| cat2_airline_multi_009 | 2 | ✓ PASS | ✓ PASS | N/A | common_prefix=12, blocks_a=14, blocks_b=14, rest_differ=True |
| cat3_000 | 3 | ✓ PASS | ✓ PASS | ✓ PASS | common_prefix=0, blocks_a=12, blocks_b=12, rest_differ=True |
| cat3_001 | 3 | ✓ PASS | ✓ PASS | ✓ PASS | common_prefix=0, blocks_a=12, blocks_b=12, rest_differ=True |
| cat3_002 | 3 | ✓ PASS | ✓ PASS | ✓ PASS | common_prefix=0, blocks_a=12, blocks_b=12, rest_differ=True |
| cat3_003 | 3 | ✓ PASS | ✓ PASS | ✓ PASS | common_prefix=0, blocks_a=11, blocks_b=11, rest_differ=True |
| cat3_004 | 3 | ✓ PASS | ✓ PASS | ✓ PASS | common_prefix=0, blocks_a=12, blocks_b=12, rest_differ=True |
| cat3_005 | 3 | ✓ PASS | ✓ PASS | ✓ PASS | common_prefix=0, blocks_a=10, blocks_b=10, rest_differ=True |
| cat3_006 | 3 | ✓ PASS | ✓ PASS | ✓ PASS | common_prefix=0, blocks_a=10, blocks_b=10, rest_differ=True |
| cat3_007 | 3 | ✓ PASS | ✓ PASS | ✓ PASS | common_prefix=0, blocks_a=10, blocks_b=10, rest_differ=True |
| cat3_008 | 3 | ✓ PASS | ✓ PASS | ✓ PASS | common_prefix=0, blocks_a=10, blocks_b=10, rest_differ=True |
| cat3_009 | 3 | ✓ PASS | ✓ PASS | ✓ PASS | common_prefix=0, blocks_a=10, blocks_b=10, rest_differ=True |
| cat4_000 | 4 | ✓ PASS | ✓ PASS | ✓ PASS | common_prefix=0, blocks_a=12, blocks_b=12, rest_differ=True |
| cat4_001 | 4 | ✓ PASS | ✓ PASS | ✓ PASS | common_prefix=0, blocks_a=12, blocks_b=12, rest_differ=True |
| cat4_002 | 4 | ✓ PASS | ✓ PASS | ✓ PASS | common_prefix=0, blocks_a=12, blocks_b=12, rest_differ=True |
| cat4_003 | 4 | ✓ PASS | ✓ PASS | ✓ PASS | common_prefix=0, blocks_a=11, blocks_b=11, rest_differ=True |
| cat4_004 | 4 | ✓ PASS | ✓ PASS | ✓ PASS | common_prefix=0, blocks_a=12, blocks_b=12, rest_differ=True |
| cat4_005 | 4 | ✓ PASS | ✓ PASS | ✓ PASS | common_prefix=0, blocks_a=10, blocks_b=10, rest_differ=True |
| cat4_006 | 4 | ✓ PASS | ✓ PASS | ✓ PASS | common_prefix=0, blocks_a=10, blocks_b=10, rest_differ=True |
| cat4_007 | 4 | ✓ PASS | ✓ PASS | ✓ PASS | common_prefix=0, blocks_a=10, blocks_b=10, rest_differ=True |
| cat4_008 | 4 | ✓ PASS | ✓ PASS | ✓ PASS | common_prefix=0, blocks_a=10, blocks_b=10, rest_differ=True |
| cat4_009 | 4 | ✓ PASS | ✓ PASS | ✓ PASS | common_prefix=0, blocks_a=10, blocks_b=10, rest_differ=True |
| cat5_000 | 5 | ✓ PASS | ✓ PASS | N/A | turns=4, actual_incremental_sharing=False, expected=False, shared_prefix_lens=[10, 12, 15] |
| cat5_001 | 5 | ✓ PASS | ✓ PASS | N/A | turns=4, actual_incremental_sharing=False, expected=False, shared_prefix_lens=[11, 13, 14] |
| cat5_002 | 5 | ✓ PASS | ✓ PASS | N/A | turns=4, actual_incremental_sharing=False, expected=False, shared_prefix_lens=[11, 12, 15] |
| cat5_003 | 5 | ✓ PASS | ✓ PASS | N/A | turns=4, actual_incremental_sharing=False, expected=False, shared_prefix_lens=[11, 13, 15] |
| cat5_004 | 5 | ✓ PASS | ✓ PASS | N/A | turns=4, actual_incremental_sharing=False, expected=False, shared_prefix_lens=[10, 12, 18] |
| cat5_005 | 5 | ✓ PASS | ✓ PASS | N/A | turns=4, actual_incremental_sharing=False, expected=False, shared_prefix_lens=[9, 11, 13] |
| cat5_006 | 5 | ✓ PASS | ✓ PASS | N/A | turns=4, actual_incremental_sharing=False, expected=False, shared_prefix_lens=[9, 11, 13] |
| cat5_007 | 5 | ✓ PASS | ✓ PASS | N/A | turns=4, actual_incremental_sharing=False, expected=False, shared_prefix_lens=[9, 11, 13] |
| cat5_008 | 5 | ✓ PASS | ✓ PASS | N/A | turns=4, actual_incremental_sharing=False, expected=False, shared_prefix_lens=[9, 12, 14] |
| cat5_009 | 5 | ✓ PASS | ✓ PASS | N/A | turns=4, actual_incremental_sharing=False, expected=False, shared_prefix_lens=[9, 11, 13] |
| cat6_000 | 6 | ✓ PASS | ✓ PASS | N/A | common_prefix=0, blocks_a=12, blocks_b=10, rest_differ=True |
| cat6_001 | 6 | ✓ PASS | ✓ PASS | N/A | common_prefix=0, blocks_a=12, blocks_b=10, rest_differ=True |
| cat6_002 | 6 | ✓ PASS | ✓ PASS | N/A | common_prefix=0, blocks_a=12, blocks_b=10, rest_differ=True |
| cat6_003 | 6 | ✓ PASS | ✓ PASS | N/A | common_prefix=0, blocks_a=11, blocks_b=10, rest_differ=True |
| cat6_004 | 6 | ✓ PASS | ✓ PASS | N/A | common_prefix=0, blocks_a=12, blocks_b=10, rest_differ=True |
| cat6_005 | 6 | ✓ PASS | ✓ PASS | N/A | common_prefix=0, blocks_a=12, blocks_b=10, rest_differ=True |
| cat6_006 | 6 | ✓ PASS | ✓ PASS | N/A | common_prefix=0, blocks_a=12, blocks_b=10, rest_differ=True |
| cat6_007 | 6 | ✓ PASS | ✓ PASS | N/A | common_prefix=0, blocks_a=12, blocks_b=10, rest_differ=True |
| cat6_008 | 6 | ✓ PASS | ✓ PASS | N/A | common_prefix=0, blocks_a=12, blocks_b=10, rest_differ=True |
| cat6_009 | 6 | ✓ PASS | ✓ PASS | N/A | common_prefix=0, blocks_a=12, blocks_b=10, rest_differ=True |

**汇总统计：**

- 用例总数: 100
- Identity check 通过: 100/100
- 父链校验通过: 100/100
- Invalidation 通过: 20/20

## 判定

- BF16 缓存恢复 KV bit-identical: ✓
- Logits max abs diff ≤ 1e-3: ✓
- Top-1 token 100% 一致: ✓
- Block identity 正确: ✓
- 父链连续性正确: ✓

**Overall: PASS**

## 正向发现：Tokenizer 非前缀稳定现象（cat5）

**核心结论**：Qwen2.5 BPE tokenizer 在 chat-template 边界（如 `\n` 与 `\nI`、`<|im_start|>assistant\n` 与紧接其后的回复首字符）会产生跨边界的合并 token。

对纯追加多轮会话（cat5）用 `apply_chat_template` 重新渲染每个前缀时，虽然文本上前缀 N+1 严格包含前缀 N，但 token id 序列并不以前缀 N 为严格前缀。分叉点通常落在追加边界（即上一轮 assistant 末尾与下一轮 user 起始的交界）附近，导致 block hash 从该点起全部不同。

- cat5 用例数: 10
- 实测 incremental_sharing=False（即 token id 前缀不复用）: 10/10
- identity_check 通过（实际行为与期望一致）: 10/10

**对 prefix caching 研究的意义**：

1. 朴素按 token id 做前缀匹配，在 chat-template 边界会丢失复用机会；
2. vLLM/HF 现行的 text-prefix-then-retokenize 方案能恢复部分复用，但牺牲了 token id 严格前缀不变性；
3. 这为 IDEA 中 C2 联合控制器需要做 boundary-aware 复用决策提供了实证依据，也支撑了 C3 "reuse value 与 fidelity 风险错位" 的核心主张。

本发现已写入 cat5 用例的 `expected_incremental_sharing=False`，作为 G0 的正向输出而非失败。
