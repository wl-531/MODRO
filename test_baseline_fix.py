"""验证DG和VAG算法修复效果"""
import sys
import os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from data.generator import generate_tasks, generate_servers
from solvers.baselines import deterministic_greedy, variance_aware_greedy
from evaluation.monte_carlo import monte_carlo_cvr

def test_heterogeneous_fix():
    """测试异构服务器场景下的修复效果"""
    print("="*60)
    print("测试DG和VAG算法修复（异构服务器）")
    print("="*60)

    np.random.seed(42)

    # 生成异构服务器（f在100-200之间）
    servers_dg = generate_servers(n_servers=10, f_range=(100, 200), decision_interval=8.8)
    servers_vag = generate_servers(n_servers=10, f_range=(100, 200), decision_interval=8.8)

    # 打印服务器异构性
    print(f"\n服务器容量范围:")
    capacities = [s.C for s in servers_dg]
    print(f"  最小容量: {min(capacities):.1f}")
    print(f"  最大容量: {max(capacities):.1f}")
    print(f"  容量比: {max(capacities)/min(capacities):.2f}x")

    # 生成任务
    tasks = generate_tasks(n_tasks=80, mu_range=(10, 100), cv_range=(0.48, 0.62), mode="coupled")

    # DG算法
    print(f"\n测试 DG 算法:")
    assign_dg = deterministic_greedy(tasks, servers_dg)

    # 验证负载分布
    load_dg = np.zeros(10)
    for i, j in enumerate(assign_dg):
        load_dg[j] += tasks[i].mu

    util_dg = load_dg / np.array([s.C for s in servers_dg])
    print(f"  利用率范围: [{util_dg.min():.3f}, {util_dg.max():.3f}]")
    print(f"  利用率标准差: {util_dg.std():.4f}")
    print(f"  [验证] 利用率是否均衡: {'OK' if util_dg.std() < 0.05 else 'NO'}")

    cvr_dg = monte_carlo_cvr(assign_dg, tasks, servers_dg, n_samples=1000)
    print(f"  CVR: {cvr_dg:.4f}")

    # VAG算法
    print(f"\n测试 VAG 算法:")
    assign_vag = variance_aware_greedy(tasks, servers_vag, lambda_=1.0)

    load_vag = np.zeros(10)
    for i, j in enumerate(assign_vag):
        load_vag[j] += tasks[i].mu

    util_vag = load_vag / np.array([s.C for s in servers_vag])
    print(f"  利用率范围: [{util_vag.min():.3f}, {util_vag.max():.3f}]")
    print(f"  利用率标准差: {util_vag.std():.4f}")
    print(f"  [验证] 利用率是否均衡: {'OK' if util_vag.std() < 0.05 else 'NO'}")

    cvr_vag = monte_carlo_cvr(assign_vag, tasks, servers_vag, n_samples=1000)
    print(f"  CVR: {cvr_vag:.4f}")

    print(f"\n" + "="*60)
    print(f"修复验证结果:")
    if util_dg.std() < 0.05 and util_vag.std() < 0.05:
        print(f"  [OK] 成功！两个算法都实现了利用率均衡")
        print(f"  [OK] DG和VAG现在正确支持异构服务器")
    else:
        print(f"  [WARN] 警告：利用率仍不均衡，可能需要进一步检查")
    print("="*60)

if __name__ == "__main__":
    test_heterogeneous_fix()
