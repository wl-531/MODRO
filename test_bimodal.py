"""验证双峰分布任务生成器"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from data.generator import generate_tasks

def test_bimodal_distribution():
    """测试双峰分布特性"""
    print("="*60)
    print("双峰分布测试")
    print("="*60)

    # 生成80个任务（与BATCH_SIZE一致）
    tasks = generate_tasks(n_tasks=80, mode="bimodal")

    # 统计任务特征
    mu_list = [t.mu for t in tasks]
    sigma_list = [t.sigma for t in tasks]
    cv_list = [t.sigma / t.mu for t in tasks]

    print(f"\n生成任务数: {len(tasks)}")
    print(f"\nμ 统计:")
    print(f"  范围: [{min(mu_list):.2f}, {max(mu_list):.2f}]")
    print(f"  均值: {np.mean(mu_list):.2f}")
    print(f"  标准差: {np.std(mu_list):.2f}")

    print(f"\nσ 统计:")
    print(f"  范围: [{min(sigma_list):.2f}, {max(sigma_list):.2f}]")
    print(f"  均值: {np.mean(sigma_list):.2f}")
    print(f"  标准差: {np.std(sigma_list):.2f}")

    print(f"\nCV 统计:")
    print(f"  范围: [{min(cv_list):.3f}, {max(cv_list):.3f}]")
    print(f"  均值: {np.mean(cv_list):.3f}")
    print(f"  标准差: {np.std(cv_list):.3f}")

    # 识别两类任务（按CV阈值0.3区分）
    type_A = [t for t in tasks if (t.sigma / t.mu) < 0.3]
    type_B = [t for t in tasks if (t.sigma / t.mu) >= 0.3]

    print(f"\n任务分类（按 CV < 0.3 区分）:")
    print(f"  Type A（大而稳）数量: {len(type_A)}")
    if type_A:
        mu_A = [t.mu for t in type_A]
        cv_A = [t.sigma / t.mu for t in type_A]
        print(f"    μ 范围: [{min(mu_A):.2f}, {max(mu_A):.2f}], 均值: {np.mean(mu_A):.2f}")
        print(f"    CV 范围: [{min(cv_A):.3f}, {max(cv_A):.3f}], 均值: {np.mean(cv_A):.3f}")

    print(f"\n  Type B（小而疯）数量: {len(type_B)}")
    if type_B:
        mu_B = [t.mu for t in type_B]
        cv_B = [t.sigma / t.mu for t in type_B]
        print(f"    μ 范围: [{min(mu_B):.2f}, {max(mu_B):.2f}], 均值: {np.mean(mu_B):.2f}")
        print(f"    CV 范围: [{min(cv_B):.3f}, {max(cv_B):.3f}], 均值: {np.mean(cv_B):.3f}")

    # 验证期望负载相近、方差差异大
    print(f"\n关键验证:")
    if type_A and type_B:
        mu_A_avg = np.mean([t.mu for t in type_A])
        mu_B_avg = np.mean([t.mu for t in type_B])
        sigma_A_avg = np.mean([t.sigma for t in type_A])
        sigma_B_avg = np.mean([t.sigma for t in type_B])

        mu_diff_ratio = abs(mu_A_avg - mu_B_avg) / max(mu_A_avg, mu_B_avg)
        sigma_diff_ratio = sigma_B_avg / sigma_A_avg

        print(f"  期望负载差异: {mu_diff_ratio*100:.1f}% (期望 < 50%)")
        print(f"  方差放大倍数: {sigma_diff_ratio:.1f}x (期望 > 4x)")
        print(f"  [验证] 双峰特性: ", end="")
        if mu_diff_ratio < 0.5 and sigma_diff_ratio > 4:
            print("成功! 期望接近但方差差异大")
        else:
            print("未达预期, 请检查参数")


def compare_modes():
    """对比耦合模式和双峰模式"""
    print("\n" + "="*60)
    print("耦合模式 vs 双峰模式对比")
    print("="*60)

    np.random.seed(42)

    # 耦合模式
    tasks_coupled = generate_tasks(n_tasks=80,
                                   mu_range=(10, 100),
                                   cv_range=(0.48, 0.62),
                                   mode="coupled")
    mu_coupled = [t.mu for t in tasks_coupled]
    cv_coupled = [t.sigma / t.mu for t in tasks_coupled]

    # 双峰模式
    tasks_bimodal = generate_tasks(n_tasks=80, mode="bimodal")
    mu_bimodal = [t.mu for t in tasks_bimodal]
    cv_bimodal = [t.sigma / t.mu for t in tasks_bimodal]

    print(f"\n耦合模式 (σ = μ × CV):")
    print(f"  μ 范围: [{min(mu_coupled):.1f}, {max(mu_coupled):.1f}]")
    print(f"  CV 范围: [{min(cv_coupled):.3f}, {max(cv_coupled):.3f}]")
    print(f"  μ-CV 相关系数: {np.corrcoef(mu_coupled, cv_coupled)[0,1]:.3f}")

    print(f"\n双峰模式 (mu-sigma 解耦):")
    print(f"  mu 范围: [{min(mu_bimodal):.1f}, {max(mu_bimodal):.1f}]")
    print(f"  CV 范围: [{min(cv_bimodal):.3f}, {max(cv_bimodal):.3f}]")
    print(f"  mu-CV 相关系数: {np.corrcoef(mu_bimodal, cv_bimodal)[0,1]:.3f}")
    print(f"  [验证] 解耦成功: ", end="")
    corr = np.corrcoef(mu_bimodal, cv_bimodal)[0,1]
    if abs(corr) < 0.3:
        print(f"mu 与 CV 弱相关 (|r| = {abs(corr):.3f} < 0.3)")
    else:
        print(f"mu 与 CV 仍有相关性 (|r| = {abs(corr):.3f})")


if __name__ == "__main__":
    np.random.seed(42)
    test_bimodal_distribution()
    compare_modes()

    print("\n" + "="*60)
    print("测试完成！")
    print("="*60)
