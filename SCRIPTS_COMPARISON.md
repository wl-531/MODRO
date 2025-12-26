# 三个主要脚本对比

## 1. experiments/run_online.py
**早期测试脚本**

```bash
python experiments/run_online.py
```

**特点：**
- 10 批次（快速测试）
- 对比 5 个算法：DG / VAG / κ-Greedy / NSGA-II / ROSA
- 输出：控制台打印 CVR 均值、O1/O2/O3 目标值
- 无 CSV 输出

**适用场景：**
- 快速验证算法是否能跑通
- 查看 ROSA 的三个目标值（O1/O2/O3）

---

## 2. experiments/run_paper_results.py
**论文实验脚本（推荐）**

```bash
python experiments/run_paper_results.py
```

**特点：**
- 50 批次（完整实验）
- Exp1 / Exp2 / Exp3 三组实验
- 对比 3 个算法：DG / NSGA / ROSA
- 输出：CSV 时序文件 + 控制台汇总

**输出文件：**
- `results_exp1_timeseries.csv`
- `results_exp2_timeseries.csv`
- `results_exp3_timeseries.csv`

**适用场景：**
- 论文实验，生成图表数据
- 对比 CVR 和 L0 时序变化

---

## 3. run_diagnostics.py
**诊断脚本**

```bash
python run_diagnostics.py
```

**特点：**
- 30 批次
- 使用 bimodal 模式（极端 μ-σ 解耦）
- 对比 4 个算法：DG / VAG / NSGA / ROSA
- 输出：`diagnostics_bimodal_extreme.csv`

**输出指标：**
- Avg|Δ|：期望 vs 实际负载波动
- Overload Count：实际超载次数

**适用场景：**
- 诊断 ROSA 性能瓶颈
- 分析实际负载波动原因

---

## 三者区别总结

| 脚本 | 批次 | 算法数 | 输出 | 用途 |
|:---|:---|:---|:---|:---|
| run_online.py | 10 | 5个 | 控制台 | 快速测试 |
| run_paper_results.py | 50 | 3个 | CSV时序 | 论文实验 |
| run_diagnostics.py | 30 | 4个 | CSV波动 | 诊断工具 |

---

## 推荐使用顺序

### 第一次运行
```bash
# 1. 论文实验（对比 CVR/L0）
python experiments/run_paper_results.py

# 2. 如果 ROSA CVR 改进小，运行诊断
python run_diagnostics.py
```

### 快速测试
```bash
# 验证代码能否运行
python experiments/run_online.py
```

---

## 总结

**想对比 CVR 和 L0？**
→ `python experiments/run_paper_results.py`（论文实验）

**想诊断波动？**
→ `python run_diagnostics.py`（诊断工具）

**想快速测试？**
→ `python experiments/run_online.py`（早期测试，可忽略）
