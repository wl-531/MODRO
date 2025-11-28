"""微批处理在线实验 - ROSA vs Baseline对比"""
import numpy as np
import sys
import os
from copy import deepcopy
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import *
from models.task import Task
from models.server import Server
from solvers.rosa import ROSASolver
from evaluation.objectives import compute_objectives
from evaluation.monte_carlo import monte_carlo_cvr
from data.generator import generate_tasks, generate_servers, validate_system_params


def deterministic_greedy(tasks, servers):
    """Deterministic-Greedy基准算法

    按期望负载E[L]=L0+Σμ最小化原则进行贪婪分配,不考虑方差σ
    """
    n_tasks = len(tasks)
    n_servers = len(servers)
    assignment = []

    current_load = np.array([s.L0 for s in servers])

    for i in range(n_tasks):
        mu_i = tasks[i].mu

        # 选择当前期望负载最小的服务器
        j_min = int(np.argmin(current_load))
        assignment.append(j_min)
        current_load[j_min] += mu_i

    return assignment


def update_server_state(servers: list, assignment: list, tasks: list,
                        processing_time: float):
    """更新服务器已有负载

    重要说明:
    - 一旦批次调度完成并经过一段时间,其负载成为 L0 的一部分
    - L0 被视为确定性负载(方差已实现或通过监控获知)
    - 因此我们不跟踪 L0 的残余方差

    简化模型: 使用期望工作量更新 L0
    完整模拟: 可改为从 N(μ, σ²) 采样实际工作量
    """
    n_servers = len(servers)

    new_load = np.zeros(n_servers)
    for i, j in enumerate(assignment):
        new_load[j] += tasks[i].mu

    for j in range(n_servers):
        processed = servers[j].f * processing_time
        servers[j].L0 = max(0.0, servers[j].L0 + new_load[j] - processed)


def run_experiment(n_batches: int = 10, verbose: bool = True):
    """运行微批处理在线实验 - ROSA vs Baseline对比"""

    # 固定随机种子确保可复现
    np.random.seed(42)

    rosa = ROSASolver(
        kappa=KAPPA, theta=THETA,
        n_pop=N_POP, g_max=G_MAX, t_max=T_MAX,
        p_c=P_C, p_m=P_M, p_risk=P_RISK, n_elite=N_ELITE,
        lambda_0=LAMBDA_0, beta=BETA,
        w1=W1, w2=W2, w3=W3
    )

    # 生成初始服务器
    servers_init = generate_servers(
        N_SERVERS,
        decision_interval=DECISION_INTERVAL
    )

    # 验证系统参数
    mu_avg = sum(MU_RANGE) / 2
    cv_avg = sum(CV_RANGE) / 2
    validation = validate_system_params(
        servers_init, BATCH_SIZE, mu_avg, cv_avg, KAPPA, THETA
    )

    print("===== 系统参数验证 =====")
    print(f"总容量: {validation['total_capacity']:.0f}")
    print(f"期望负载: {validation['expected_load']:.0f}")
    print(f"鲁棒负载(分散): {validation['robust_load']:.0f}")
    print(f"鲁棒Buffer(分散): {validation['robust_buffer']:.0f}")
    print(f"鲁棒Buffer(聚合): {validation['aggregated_buffer']:.0f}")
    print(f"Buffer放大倍数: {validation['buffer_ratio']:.2f} (理论值sqrt{N_SERVERS}={np.sqrt(N_SERVERS):.2f})")
    print(f"期望负载率 rho: {validation['rho_expected']:.3f}")
    print(f"鲁棒负载率 rho_robust: {validation['rho_robust']:.3f}")
    print(f"有效负载率 rho_eff (theta={THETA}): {validation['rho_effective']:.3f}")
    print(f"系统可行性: {'[OK] 可行' if validation['feasible'] else '[FAIL] 不可行'}")
    print()

    if not validation['feasible']:
        print("警告: 系统参数导致无可行解,请增加 DECISION_INTERVAL")
        return None

    # [修复] 使用deepcopy创建完全独立的双环境
    servers_baseline = deepcopy(servers_init)
    servers_rosa = deepcopy(servers_init)

    results_baseline = {'cvr': [], 'residual': []}
    results_rosa = {'cvr': [], 'O1': [], 'O2': [], 'O3': [], 'residual': []}

    print("===== ROSA vs Baseline 对比实验 =====\n")

    for batch_idx in range(n_batches):
        # 生成相同的任务批次
        tasks = generate_tasks(BATCH_SIZE, MU_RANGE, CV_RANGE)

        # [关键] 记录更新前的残留负载
        residual_baseline_before = sum(s.L0 for s in servers_baseline)
        residual_rosa_before = sum(s.L0 for s in servers_rosa)

        # Baseline算法
        assignment_baseline = deterministic_greedy(tasks, servers_baseline)
        cvr_baseline = monte_carlo_cvr(assignment_baseline, tasks, servers_baseline, MC_SAMPLES)

        # ROSA算法
        assignment_rosa = rosa.solve_batch(tasks, servers_rosa)
        cvr_rosa = monte_carlo_cvr(assignment_rosa, tasks, servers_rosa, MC_SAMPLES)
        obj_rosa = compute_objectives(assignment_rosa, tasks, servers_rosa, KAPPA, THETA)

        # [关键] 先更新服务器状态，再记录更新后的残留负载
        update_server_state(servers_baseline, assignment_baseline, tasks,
                           processing_time=DECISION_INTERVAL)
        update_server_state(servers_rosa, assignment_rosa, tasks,
                           processing_time=DECISION_INTERVAL)

        residual_baseline_after = sum(s.L0 for s in servers_baseline)
        residual_rosa_after = sum(s.L0 for s in servers_rosa)

        # 记录结果
        results_baseline['cvr'].append(cvr_baseline)
        results_baseline['residual'].append(residual_baseline_after)

        results_rosa['cvr'].append(cvr_rosa)
        results_rosa['O1'].append(obj_rosa['O1'])
        results_rosa['O2'].append(obj_rosa['O2'])
        results_rosa['O3'].append(obj_rosa['O3'])
        results_rosa['residual'].append(residual_rosa_after)

        # 计算改进百分比
        improvement = ((cvr_baseline - cvr_rosa) / max(cvr_baseline, 1e-6)) * 100

        if verbose:
            print(f"Batch {batch_idx + 1}/{n_batches}: "
                  f"Baseline_CVR={cvr_baseline:.4f} (L0: {residual_baseline_before:.1f}->{residual_baseline_after:.1f}) | "
                  f"ROSA_CVR={cvr_rosa:.4f} (L0: {residual_rosa_before:.1f}->{residual_rosa_after:.1f}) "
                  f"[Improvement: {improvement:+.1f}%]")

    # 汇总结果
    print("\n===== 实验结果汇总 =====")
    print(f"\n[Baseline - Deterministic Greedy]")
    print(f"  平均 CVR: {np.mean(results_baseline['cvr']):.4f} ± {np.std(results_baseline['cvr']):.4f}")
    print(f"  平均残留: {np.mean(results_baseline['residual']):.1f}")

    print(f"\n[ROSA - Robust Online Scheduling]")
    print(f"  平均 CVR: {np.mean(results_rosa['cvr']):.4f} ± {np.std(results_rosa['cvr']):.4f}")
    print(f"  平均 O1:  {np.mean(results_rosa['O1']):.2f} ± {np.std(results_rosa['O1']):.2f}")
    print(f"  平均 O2:  {np.mean(results_rosa['O2']):.4f} ± {np.std(results_rosa['O2']):.4f}")
    print(f"  平均 O3:  {np.mean(results_rosa['O3']):.2f} ± {np.std(results_rosa['O3']):.2f}")
    print(f"  平均残留: {np.mean(results_rosa['residual']):.1f}")

    # 总体改进
    avg_baseline_cvr = np.mean(results_baseline['cvr'])
    avg_rosa_cvr = np.mean(results_rosa['cvr'])
    total_improvement = ((avg_baseline_cvr - avg_rosa_cvr) / max(avg_baseline_cvr, 1e-6)) * 100

    print(f"\n[对比结果]")
    print(f"  ROSA相对Baseline的CVR改进: {total_improvement:+.1f}%")

    if avg_rosa_cvr < ALPHA:
        print(f"  ROSA性能: [成功] CVR ({avg_rosa_cvr:.4f}) < α ({ALPHA})")
    else:
        print(f"  ROSA性能: [失败] CVR ({avg_rosa_cvr:.4f}) >= α ({ALPHA})")

    return {
        'baseline': results_baseline,
        'rosa': results_rosa
    }


if __name__ == '__main__':
    results = run_experiment(n_batches=10, verbose=True)
