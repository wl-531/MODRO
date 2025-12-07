"""
ICDCS 论文实验脚本
实验1: 主对比 | 实验2: CV敏感性 | 实验3: 消融实验

支持4个算法的完整对比：DG / VAG / NSGA2Mean / ROSA
"""
import sys
import os
import numpy as np
import pandas as pd
from copy import deepcopy

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
from data.generator import generate_tasks, generate_servers
from solvers.rosa import ROSASolver
# NEW: 导入新baseline算法
from solvers.baselines import deterministic_greedy, variance_aware_greedy
from solvers.nsga2_mean import NSGA2MeanSolver
from experiments.run_online import update_server_state
from evaluation.monte_carlo import monte_carlo_cvr

def set_config(updates: dict):
    for key, val in updates.items():
        setattr(config, key, val)

# NEW: 添加VAG_LAMBDA到config（如果不存在）
if not hasattr(config, 'VAG_LAMBDA'):
    setattr(config, 'VAG_LAMBDA', 1.0)

def run_batch_simulation(n_batches=50, label="Experiment", enable_all_algos=True,
                         task_mode="coupled"):
    """运行批量仿真，支持4个算法：DG / VAG / NSGA2Mean / ROSA

    Args:
        n_batches: 批次数量
        label: 实验标签
        enable_all_algos: 是否启用所有算法（False时只运行DG和ROSA，兼容旧版）
        task_mode: 任务生成模式
            - "coupled": 原模式，σ = μ × CV（默认）
            - "bimodal": 双峰模式，μ-σ 解耦

    Returns:
        results: {algo_name: {'cvr': [...], 'L0': [...]}}
    """
    print(f"\n>>> Running {label} (n={n_batches}, task_mode={task_mode})...")
    np.random.seed(42)

    # MODIFIED: 创建4份独立服务器副本
    servers_init = generate_servers(config.N_SERVERS, decision_interval=config.DECISION_INTERVAL)
    servers_dg = deepcopy(servers_init)
    servers_rosa = deepcopy(servers_init)

    if enable_all_algos:  # NEW
        servers_vag = deepcopy(servers_init)
        servers_nsga = deepcopy(servers_init)

    # MODIFIED: 初始化求解器
    rosa = ROSASolver(
        kappa=config.KAPPA, theta=config.THETA,
        n_pop=config.N_POP, g_max=config.G_MAX, t_max=config.T_MAX,
        p_c=config.P_C, p_m=config.P_M, p_risk=config.P_RISK,
        n_elite=config.N_ELITE, lambda_0=config.LAMBDA_0, beta=config.BETA,
        w1=config.W1, w2=config.W2, w3=config.W3
    )

    if enable_all_algos:  # NEW: NSGA2Mean求解器
        nsga = NSGA2MeanSolver(
            n_pop=config.N_POP, g_max=config.G_MAX, t_max=config.T_MAX,
            p_c=config.P_C, p_m=config.P_M, n_elite=config.N_ELITE,
            w1=config.W1, w2=config.W2, w3=config.W3
        )

    # MODIFIED: 结果记录结构（按算法组织）
    results = {
        'dg': {'cvr': [], 'L0': []},
        'rosa': {'cvr': [], 'L0': []}
    }
    if enable_all_algos:  # NEW
        results['vag'] = {'cvr': [], 'L0': []}
        results['nsga'] = {'cvr': [], 'L0': []}

    for i in range(n_batches):
        # MODIFIED: 传入 task_mode 参数
        tasks = generate_tasks(config.BATCH_SIZE, config.MU_RANGE, config.CV_RANGE,
                               mode=task_mode)

        # DG算法
        assign_dg = deterministic_greedy(tasks, servers_dg)
        cvr_dg = monte_carlo_cvr(assign_dg, tasks, servers_dg, config.MC_SAMPLES)
        L0_dg = sum(s.L0 for s in servers_dg)
        update_server_state(servers_dg, assign_dg, tasks, config.DECISION_INTERVAL)

        # ROSA算法
        assign_rosa = rosa.solve_batch(tasks, servers_rosa)
        cvr_rosa = monte_carlo_cvr(assign_rosa, tasks, servers_rosa, config.MC_SAMPLES)
        L0_rosa = sum(s.L0 for s in servers_rosa)
        update_server_state(servers_rosa, assign_rosa, tasks, config.DECISION_INTERVAL)

        # NEW: VAG算法
        if enable_all_algos:
            assign_vag = variance_aware_greedy(tasks, servers_vag, lambda_=config.VAG_LAMBDA)
            cvr_vag = monte_carlo_cvr(assign_vag, tasks, servers_vag, config.MC_SAMPLES)
            L0_vag = sum(s.L0 for s in servers_vag)
            update_server_state(servers_vag, assign_vag, tasks, config.DECISION_INTERVAL)

        # NEW: NSGA2Mean算法
        if enable_all_algos:
            assign_nsga = nsga.solve_batch(tasks, servers_nsga)
            cvr_nsga = monte_carlo_cvr(assign_nsga, tasks, servers_nsga, config.MC_SAMPLES)
            L0_nsga = sum(s.L0 for s in servers_nsga)
            update_server_state(servers_nsga, assign_nsga, tasks, config.DECISION_INTERVAL)

        # MODIFIED: 记录结果
        results['dg']['cvr'].append(cvr_dg)
        results['dg']['L0'].append(L0_dg)
        results['rosa']['cvr'].append(cvr_rosa)
        results['rosa']['L0'].append(L0_rosa)

        if enable_all_algos:  # NEW
            results['vag']['cvr'].append(cvr_vag)
            results['vag']['L0'].append(L0_vag)
            results['nsga']['cvr'].append(cvr_nsga)
            results['nsga']['L0'].append(L0_nsga)

        # MODIFIED: 打印进度
        if enable_all_algos:
            print(f"\r  Batch {i+1}/{n_batches} | DG={cvr_dg:.4f} VAG={cvr_vag:.4f} NSGA={cvr_nsga:.4f} ROSA={cvr_rosa:.4f}", end="")
        else:
            print(f"\r  Batch {i+1}/{n_batches} | DG CVR={cvr_dg:.4f} L0={L0_dg:.1f} | ROSA CVR={cvr_rosa:.4f} L0={L0_rosa:.1f}", end="")

    print()
    return results

def exp1_main_comparison():
    """实验1: 主对比实验（4算法）"""
    print("\n" + "="*60)
    print("EXP 1: Main Comparison (DG / VAG / NSGA2Mean / ROSA)")
    print("="*60)

    set_config({
        'BATCH_SIZE': 80,
        'DECISION_INTERVAL': 8.8,
        'CV_RANGE': (0.48, 0.62),
        'ALPHA': 0.15,
        'KAPPA': 2.38,
        'W1': 0.40, 'W2': 0.25, 'W3': 0.35,
        'VAG_LAMBDA': 1.0  # NEW
    })

    # MODIFIED: 运行4个算法
    res = run_batch_simulation(n_batches=50, label="Main_Comparison", enable_all_algos=True)

    # MODIFIED: 统计所有算法结果
    print(f"\n[Results]")
    for algo_name in ['dg', 'vag', 'nsga', 'rosa']:
        cvr_mean = np.mean(res[algo_name]['cvr'])
        cvr_std = np.std(res[algo_name]['cvr'])
        L0_mean = np.mean(res[algo_name]['L0'])
        print(f"  {algo_name.upper():8s}: CVR = {cvr_mean:.4f} ± {cvr_std:.4f}, Avg L0 = {L0_mean:.1f}")

    # MODIFIED: 计算相对DG的改进
    dg_cvr = np.mean(res['dg']['cvr'])
    print(f"\n[CVR Reduction vs DG]")
    for algo in ['vag', 'nsga', 'rosa']:
        improvement = (dg_cvr - np.mean(res[algo]['cvr'])) / dg_cvr * 100
        print(f"  {algo.upper():8s}: {improvement:+.1f}%")

    # NEW: 保存到CSV
    csv_data = []
    for algo_name in ['dg', 'vag', 'nsga', 'rosa']:
        csv_data.append({
            'exp_type': 'main',
            'algo': algo_name,
            'cv_mean': 0.55,  # (0.48+0.62)/2
            'cvr_mean': np.mean(res[algo_name]['cvr']),
            'cvr_std': np.std(res[algo_name]['cvr']),
            'L0_mean': np.mean(res[algo_name]['L0']),
            'cvr_max': np.max(res[algo_name]['cvr'])
        })

    df = pd.DataFrame(csv_data)
    df.to_csv("results_exp1_main_comparison.csv", index=False)
    print(f"\nSaved to results_exp1_main_comparison.csv")

    return res

def exp2_sensitivity_cv():
    """实验2: CV敏感性分析（4算法）"""
    print("\n" + "="*60)
    print("EXP 2: Sensitivity to CV (Uncertainty Level)")
    print("="*60)

    cv_means = [0.25, 0.40, 0.55, 0.70, 0.85]
    data_points = []

    for mean_cv in cv_means:
        cv_range = (mean_cv - 0.07, mean_cv + 0.07)
        set_config({
            'CV_RANGE': cv_range,
            'W1': 0.40, 'W2': 0.25, 'W3': 0.35,
            'VAG_LAMBDA': 1.0
        })

        # MODIFIED: 运行4个算法
        res = run_batch_simulation(n_batches=30, label=f"CV={mean_cv}", enable_all_algos=True)

        # MODIFIED: 记录所有算法结果
        for algo_name in ['dg', 'vag', 'nsga', 'rosa']:
            data_points.append({
                'exp_type': 'cv_sensitivity',
                'algo': algo_name,
                'cv_mean': mean_cv,
                'cvr_mean': np.mean(res[algo_name]['cvr']),
                'cvr_std': np.std(res[algo_name]['cvr']),
                'L0_mean': np.mean(res[algo_name]['L0']),
                'cvr_max': np.max(res[algo_name]['cvr'])
            })

        # MODIFIED: 打印所有算法结果
        print(f"  CV={mean_cv}: DG={np.mean(res['dg']['cvr']):.4f}, "
              f"VAG={np.mean(res['vag']['cvr']):.4f}, "
              f"NSGA={np.mean(res['nsga']['cvr']):.4f}, "
              f"ROSA={np.mean(res['rosa']['cvr']):.4f}")

    df = pd.DataFrame(data_points)
    df.to_csv("results_exp2_cv_sensitivity.csv", index=False)
    print(f"\nSaved to results_exp2_cv_sensitivity.csv")
    return df

def exp3_ablation_study():
    """实验3: 消融实验 - 高压环境（4算法 + ROSA消融）"""
    print("\n" + "="*60)
    print("EXP 3: Ablation Study (O3 Risk Term) - High Pressure")
    print("="*60)

    # 关键：使用高压配置，让O3的价值显现
    set_config({
        'BATCH_SIZE': 80,
        'DECISION_INTERVAL': 8.0,    # 缩短间隔，增加压力
        'CV_RANGE': (0.60, 0.80),    # 增大波动
        'ALPHA': 0.15,
        'KAPPA': 2.38,
        'THETA': 1.0,
        'VAG_LAMBDA': 1.0
    })

    # MODIFIED: 先运行所有baseline算法作为参照
    print("\nRunning all algorithms under high pressure (W3=0.35)...")
    set_config({'W1': 0.40, 'W2': 0.25, 'W3': 0.35})
    res_full = run_batch_simulation(n_batches=30, label="All_Algos", enable_all_algos=True)

    # ROSA-NoRisk（W3=0消融版本）
    print("\nRunning ROSA-NoRisk (W3=0) under high pressure...")
    set_config({'W1': 0.40, 'W2': 0.25, 'W3': 0.0})
    res_norisk = run_batch_simulation(n_batches=30, label="ROSA-NoRisk", enable_all_algos=False)

    # MODIFIED: 汇总结果
    cvr_dg = np.mean(res_full['dg']['cvr'])
    cvr_vag = np.mean(res_full['vag']['cvr'])
    cvr_nsga = np.mean(res_full['nsga']['cvr'])
    cvr_rosa_full = np.mean(res_full['rosa']['cvr'])
    cvr_rosa_norisk = np.mean(res_norisk['rosa']['cvr'])

    print(f"\n[Ablation Results - High Pressure]")
    print(f"  DG:          CVR = {cvr_dg:.4f}")
    print(f"  VAG:         CVR = {cvr_vag:.4f}")
    print(f"  NSGA-Mean:   CVR = {cvr_nsga:.4f}")
    print(f"  ROSA-Full:   CVR = {cvr_rosa_full:.4f}")
    print(f"  ROSA-NoRisk: CVR = {cvr_rosa_norisk:.4f}")

    if cvr_rosa_norisk > cvr_rosa_full and cvr_rosa_norisk > 0:
        reduction = (cvr_rosa_norisk - cvr_rosa_full) / cvr_rosa_norisk * 100
        print(f"\n  O3 reduces CVR by {reduction:.1f}%")
    else:
        print(f"\n  Warning: O3 effect not observed, may need more pressure")

    # NEW: 保存到CSV
    csv_data = []
    for algo_name in ['dg', 'vag', 'nsga', 'rosa']:
        csv_data.append({
            'exp_type': 'ablation',
            'algo': algo_name,
            'variant': 'full' if algo_name == 'rosa' else 'baseline',
            'cv_mean': 0.70,  # (0.60+0.80)/2
            'cvr_mean': np.mean(res_full[algo_name]['cvr']),
            'cvr_std': np.std(res_full[algo_name]['cvr']),
            'L0_mean': np.mean(res_full[algo_name]['L0']),
            'cvr_max': np.max(res_full[algo_name]['cvr'])
        })

    # 添加ROSA-NoRisk的结果
    csv_data.append({
        'exp_type': 'ablation',
        'algo': 'rosa',
        'variant': 'norisk',
        'cv_mean': 0.70,
        'cvr_mean': cvr_rosa_norisk,
        'cvr_std': np.std(res_norisk['rosa']['cvr']),
        'L0_mean': np.mean(res_norisk['rosa']['L0']),
        'cvr_max': np.max(res_norisk['rosa']['cvr'])
    })

    df = pd.DataFrame(csv_data)
    df.to_csv("results_exp3_ablation.csv", index=False)
    print(f"\nSaved to results_exp3_ablation.csv")

    return {
        'dg': cvr_dg,
        'vag': cvr_vag,
        'nsga': cvr_nsga,
        'rosa_full': cvr_rosa_full,
        'rosa_norisk': cvr_rosa_norisk
    }

def exp4_bimodal_comparison():
    """实验4: 双峰分布对比实验（验证 μ-σ 解耦后 ROSA 优势）"""
    print("\n" + "="*60)
    print("EXP 4: Bimodal Distribution (μ-σ Decoupled)")
    print("="*60)

    # 使用与 exp1 相同的基础配置
    set_config({
        'BATCH_SIZE': 80,
        'DECISION_INTERVAL': 8.8,
        'ALPHA': 0.15,
        'KAPPA': 2.38,
        'W1': 0.40, 'W2': 0.25, 'W3': 0.35,
        'VAG_LAMBDA': 1.0
    })

    # 关键：使用 task_mode="bimodal"
    res = run_batch_simulation(n_batches=50, label="Bimodal_Comparison",
                               enable_all_algos=True,
                               task_mode="bimodal")

    # 统计所有算法结果
    print(f"\n[Results - Bimodal Distribution]")
    for algo_name in ['dg', 'vag', 'nsga', 'rosa']:
        cvr_mean = np.mean(res[algo_name]['cvr'])
        cvr_std = np.std(res[algo_name]['cvr'])
        L0_mean = np.mean(res[algo_name]['L0'])
        print(f"  {algo_name.upper():8s}: CVR = {cvr_mean:.4f} ± {cvr_std:.4f}, Avg L0 = {L0_mean:.1f}")

    # 计算相对DG的改进
    dg_cvr = np.mean(res['dg']['cvr'])
    print(f"\n[CVR Reduction vs DG]")
    for algo in ['vag', 'nsga', 'rosa']:
        improvement = (dg_cvr - np.mean(res[algo]['cvr'])) / dg_cvr * 100
        print(f"  {algo.upper():8s}: {improvement:+.1f}%")

    # 保存到 CSV
    csv_data = []
    for algo_name in ['dg', 'vag', 'nsga', 'rosa']:
        csv_data.append({
            'exp_type': 'bimodal',
            'algo': algo_name,
            'cv_mean': np.nan,  # 双峰模式无单一 CV 值，用 NaN 占位
            'cvr_mean': np.mean(res[algo_name]['cvr']),
            'cvr_std': np.std(res[algo_name]['cvr']),
            'L0_mean': np.mean(res[algo_name]['L0']),
            'cvr_max': np.max(res[algo_name]['cvr'])
        })

    df = pd.DataFrame(csv_data)
    df.to_csv("results_exp4_bimodal.csv", index=False)
    print(f"\nSaved to results_exp4_bimodal.csv")

    return res

if __name__ == "__main__":
    print("="*60)
    print("ICDCS Paper Experiments")
    print("="*60)

    #exp1_main_comparison()
    #exp2_sensitivity_cv()
    #exp3_ablation_study()
    exp4_bimodal_comparison()

    print("\n" + "="*60)
    print("All experiments completed!")
    print("="*60)
