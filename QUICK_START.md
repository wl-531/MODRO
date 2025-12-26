# 快速开始指南

## 运行哪个文件？

### 1. 论文实验（对比 CVR 和 L0）★ 推荐
```bash
python experiments/run_paper_results.py
```

**输出：**
- `results_exp1_timeseries.csv` - CVR 和 L0 时序数据
- `results_exp2_timeseries.csv`
- `results_exp3_timeseries.csv`

**特点：**
- 50 批次，完整实验
- Exp1 / Exp2 / Exp3 三组对比
- 对比 DG / NSGA / ROSA

---

### 2. 诊断工具（期望 vs 实际波动）
```bash
python run_diagnostics.py
```

**输出：**
- `diagnostics_bimodal_extreme.csv` - 服务器波动详细数据

**特点：**
- 30 批次，bimodal 模式
- 诊断 ROSA 是否真的降低波动
- 对比 DG / VAG / NSGA / ROSA

---

### 3. 快速测试（可选）
```bash
python experiments/run_online.py
```

**输出：**
- 仅控制台打印

**特点：**
- 10 批次，快速验证
- 对比 5 个算法
- 查看 O1/O2/O3 目标值

---

## 两者区别

| 脚本 | 目的 | 关注点 | 运行时间 |
|:---|:---|:---|:---|
| `run_paper_results.py` | 论文实验，对比算法性能 | CVR 改进幅度，L0 累积 | ~20分钟 |
| `run_diagnostics.py` | 诊断 ROSA 性能瓶颈 | 实际负载波动，风险分散 | ~10分钟 |

---

## 推荐流程

1. **先运行论文实验**（如果还没跑过）
   ```bash
   python experiments/run_paper_results.py
   ```

2. **如果 ROSA 的 CVR 改进不明显**，运行诊断
   ```bash
   python run_diagnostics.py
   ```

3. **分析诊断结果**
   - 看 ROSA 的 Avg|Δ| 是否比 DG 低
   - 如果不低，说明 ROSA 没有真正降低波动

---

## 文件目录

```
experiments/
  run_paper_results.py      ← 运行这个对比 CVR/L0

run_diagnostics.py          ← 运行这个诊断波动

results_exp1_timeseries.csv ← CVR/L0 数据
diagnostics_bimodal_extreme.csv ← 波动数据
```

---

## 总结

**想看 CVR 和 L0？**
→ `python experiments/run_paper_results.py`

**想看期望 vs 实际波动？**
→ `python run_diagnostics.py`
