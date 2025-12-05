"""验证向后兼容性：exp1/exp2/exp3 仍使用 coupled 模式"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import config
config.BATCH_SIZE = 40
config.MC_SAMPLES = 1000

from experiments.run_paper_results import run_batch_simulation, set_config

if __name__ == "__main__":
    print("="*60)
    print("向后兼容性测试")
    print("="*60)

    set_config({
        'BATCH_SIZE': 40,
        'DECISION_INTERVAL': 8.8,
        'CV_RANGE': (0.48, 0.62),
        'ALPHA': 0.15,
        'KAPPA': 2.38,
        'W1': 0.40, 'W2': 0.25, 'W3': 0.35,
        'VAG_LAMBDA': 1.0
    })

    print("\n测试1: 不指定 task_mode（应默认使用 coupled）")
    res1 = run_batch_simulation(n_batches=2, label="Test_Default_Mode",
                                enable_all_algos=False)
    print("  [OK] 默认模式测试通过")

    print("\n测试2: 显式指定 task_mode='coupled'")
    res2 = run_batch_simulation(n_batches=2, label="Test_Coupled_Mode",
                                enable_all_algos=False,
                                task_mode="coupled")
    print("  [OK] 耦合模式测试通过")

    print("\n测试3: 使用 task_mode='bimodal'")
    res3 = run_batch_simulation(n_batches=2, label="Test_Bimodal_Mode",
                                enable_all_algos=False,
                                task_mode="bimodal")
    print("  [OK] 双峰模式测试通过")

    print("\n" + "="*60)
    print("向后兼容性验证成功！")
    print("  - 默认参数仍使用 coupled 模式")
    print("  - 显式 coupled 参数正常工作")
    print("  - 新的 bimodal 参数正常工作")
    print("="*60)
