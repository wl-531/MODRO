# 服务器监控诊断 - 使用指南

## 只需运行一个文件

```bash
python run_diagnostics.py
```

**这个脚本会：**
1. 运行 30 批次实验
2. 使用 bimodal 模式（陷阱任务 vs 稳定任务）
3. 对比 4 个算法：DG / VAG / NSGA / ROSA
4. 生成诊断文件：`diagnostics_bimodal_extreme.csv`

## 输出结果

### 控制台输出
```
[期望 vs 实际 波动分析]
Algo  | Avg|Δ| | Max|Δ| | Std(Δ) | Overload Count
------|---------|--------|--------|---------------
DG    |   573.7 | 1901.5 |  403.5 |            300
ROSA  |   550.8 | 1715.0 |  423.6 |            300
...
```

**关键指标：**
- `Avg|Δ|`: 平均波动，越小越好
- `Overload Count`: 超载次数

### CSV 文件
- 文件名：`diagnostics_bimodal_extreme.csv`
- 内容：每个批次、每台服务器的详细状态
- 列：batch_idx, algo, server_id, capacity, mu_j, sigma_j, actual_load, delta, overload

## 配置说明

在 `run_diagnostics.py` 中可以调整：

```python
set_config({
    'N_SERVERS': 10,        # 服务器数量
    'BATCH_SIZE': 500,      # 每批任务数
    'DECISION_INTERVAL': 25, # 决策周期
    # ...
})

run_batch_simulation(
    n_batches=30,           # 批次数
    task_mode="bimodal",    # 任务模式
    # ...
)
```

### 任务模式选项
- `"bimodal"`: 极端解耦（陷阱任务 vs 稳定任务）
- `"multiclass"`: 多类型任务（视频/IO/语音/后台）
- `"coupled"`: 传统模式（σ = μ × CV）

## 其他文件说明

| 文件 | 作用 |
|:---|:---|
| `run_diagnostics.py` | **主脚本，运行这个** |
| `evaluation/server_monitor.py` | 监控模块代码 |
| `diagnostics_bimodal_extreme.csv` | 运行后生成的数据文件 |
| `BIMODAL_TEST_RESULTS.md` | 测试结果总结 |

## 快速开始

```bash
# 1. 运行诊断
python run_diagnostics.py

# 2. 查看汇总
#    自动打印在控制台

# 3. 分析详细数据
import pandas as pd
df = pd.read_csv('diagnostics_bimodal_extreme.csv')
df.groupby('algo')['delta'].agg(['mean', 'std', 'max'])
```

## 完成！

就这么简单，其他文件都不用管。
