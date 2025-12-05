"""快速测试 exp4_bimodal_comparison 函数"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# 临时降低批次数，加快测试
import config
config.BATCH_SIZE = 40  # 降低到40
config.MC_SAMPLES = 1000  # 降低到1000

from experiments.run_paper_results import exp4_bimodal_comparison

if __name__ == "__main__":
    print("="*60)
    print("测试 exp4_bimodal_comparison (小规模)")
    print("="*60)

    # 测试运行（仅3批次）
    import experiments.run_paper_results as exp_module

    # 临时修改 run_batch_simulation 内部的批次数
    def quick_exp4():
        """快速版本的 exp4，只运行3批次"""
        from experiments.run_paper_results import run_batch_simulation, set_config
        import numpy as np
        import pandas as pd

        print("\n" + "="*60)
        print("EXP 4: Bimodal Distribution (μ-σ Decoupled) - Quick Test")
        print("="*60)

        set_config({
            'BATCH_SIZE': 40,
            'DECISION_INTERVAL': 8.8,
            'ALPHA': 0.15,
            'KAPPA': 2.38,
            'W1': 0.40, 'W2': 0.25, 'W3': 0.35,
            'VAG_LAMBDA': 1.0
        })

        # 关键：使用 task_mode="bimodal"，只运行3批次
        res = run_batch_simulation(n_batches=3, label="Bimodal_Quick_Test",
                                   enable_all_algos=True,
                                   task_mode="bimodal")

        print(f"\n[Results - Bimodal Distribution (3 batches)]")
        for algo_name in ['dg', 'vag', 'nsga', 'rosa']:
            cvr_mean = np.mean(res[algo_name]['cvr'])
            cvr_std = np.std(res[algo_name]['cvr'])
            print(f"  {algo_name.upper():8s}: CVR = {cvr_mean:.4f} ± {cvr_std:.4f}")

        print("\n[验证] 测试成功！函数可正常运行。")
        return res

    quick_exp4()
