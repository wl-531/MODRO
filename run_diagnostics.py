"""测试修正后的服务器监控模块 - 期望 vs 实际负载对比"""
import sys
sys.path.insert(0, '.')

import config
from experiments.run_paper_results import run_batch_simulation, set_config

# 简化配置，快速测试
set_config({
    'N_SERVERS': 10,
    'BATCH_SIZE': 425,
    'DECISION_INTERVAL': 25,
    'ALPHA': 0.15,
    'KAPPA': 2.38,
    'THETA': 1.0,
    'W1': 0.40, 'W2': 0.25, 'W3': 0.35,
    'VAG_LAMBDA': 1.0,
    'N_POP': 50,    # 减小种群
    'G_MAX': 50,    # 减小迭代
    'T_MAX': 1
})

print("="*60)
print("测试极端 mu-sigma 解耦 (Bimodal 模式)")
print("="*60)

# 运行实验，使用 bimodal 模式测试陷阱任务
res = run_batch_simulation(
    n_batches=30,
    label="Bimodal_Extreme",
    enable_all_algos=True,
    task_mode="bimodal",  # 极端解耦模式
    enable_diagnostics=True
)

print("\n测试完成！")
print("检查生成的文件：diagnostics_bimodal_extreme.csv")
print("\n预期效果：")
print("- DG 会选择低mu任务(陷阱), 导致实际负载波动大")
print("- ROSA 应该识别高sigma风险, 分散陷阱任务, 波动更小")
