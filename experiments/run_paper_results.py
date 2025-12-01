"""
ICDCS 论文实验脚本
实验1: 主对比 | 实验2: CV敏感性 | 实验3: 消融实验
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
from experiments.run_online import deterministic_greedy, update_server_state
from evaluation.monte_carlo import monte_carlo_cvr

def set_config(updates: dict):
    for key, val in updates.items():
        setattr(config, key, val)

def run_batch_simulation(n_batches=50, label="Experiment"):
    print(f"\n>>> Running {label} (n={n_batches})...")
    np.random.seed(42)

    servers_init = generate_servers(config.N_SERVERS, decision_interval=config.DECISION_INTERVAL)
    servers_baseline = deepcopy(servers_init)
    servers_rosa = deepcopy(servers_init)

    rosa = ROSASolver(
        kappa=config.KAPPA, theta=config.THETA,
        n_pop=config.N_POP, g_max=config.G_MAX, t_max=config.T_MAX,
        p_c=config.P_C, p_m=config.P_M, p_risk=config.P_RISK,
        n_elite=config.N_ELITE, lambda_0=config.LAMBDA_0, beta=config.BETA,
        w1=config.W1, w2=config.W2, w3=config.W3
    )

    results = {'baseline_cvr': [], 'rosa_cvr': [], 'baseline_L0': [], 'rosa_L0': []}

    for i in range(n_batches):
        tasks = generate_tasks(config.BATCH_SIZE, config.MU_RANGE, config.CV_RANGE)

        assign_base = deterministic_greedy(tasks, servers_baseline)
        cvr_base = monte_carlo_cvr(assign_base, tasks, servers_baseline, config.MC_SAMPLES)
        L0_base_before = sum(s.L0 for s in servers_baseline)
        update_server_state(servers_baseline, assign_base, tasks, config.DECISION_INTERVAL)

        assign_rosa = rosa.solve_batch(tasks, servers_rosa)
        cvr_rosa = monte_carlo_cvr(assign_rosa, tasks, servers_rosa, config.MC_SAMPLES)
        L0_rosa_before = sum(s.L0 for s in servers_rosa)
        update_server_state(servers_rosa, assign_rosa, tasks, config.DECISION_INTERVAL)

        results['baseline_cvr'].append(cvr_base)
        results['rosa_cvr'].append(cvr_rosa)
        results['baseline_L0'].append(L0_base_before)
        results['rosa_L0'].append(L0_rosa_before)

        print(f"\r  Batch {i+1}/{n_batches} | Base CVR={cvr_base:.4f} L0={L0_base_before:.1f} | ROSA CVR={cvr_rosa:.4f} L0={L0_rosa_before:.1f}", end="")

    print()
    return results

def exp1_main_comparison():
    """实验1: 主对比实验"""
    print("\n" + "="*60)
    print("EXP 1: Main Comparison")
    print("="*60)

    set_config({
        'BATCH_SIZE': 80,
        'DECISION_INTERVAL': 8.8,
        'CV_RANGE': (0.48, 0.62),
        'ALPHA': 0.15,
        'KAPPA': 2.38,
        'W1': 0.40, 'W2': 0.25, 'W3': 0.35
    })

    res = run_batch_simulation(n_batches=50, label="Main_Comparison")

    print(f"\n[Results]")
    print(f"  Baseline: CVR = {np.mean(res['baseline_cvr']):.4f} ± {np.std(res['baseline_cvr']):.4f}")
    print(f"  ROSA:     CVR = {np.mean(res['rosa_cvr']):.4f} ± {np.std(res['rosa_cvr']):.4f}")
    print(f"  Worst Batch - Baseline: {np.max(res['baseline_cvr']):.4f}, ROSA: {np.max(res['rosa_cvr']):.4f}")
    print(f"  Avg L0 - Baseline: {np.mean(res['baseline_L0']):.1f}, ROSA: {np.mean(res['rosa_L0']):.1f}")

    improvement = (np.mean(res['baseline_cvr']) - np.mean(res['rosa_cvr'])) / np.mean(res['baseline_cvr']) * 100
    print(f"  CVR Reduction: {improvement:.1f}%")

    return res

def exp2_sensitivity_cv():
    """实验2: CV敏感性分析"""
    print("\n" + "="*60)
    print("EXP 2: Sensitivity to CV (Uncertainty Level)")
    print("="*60)

    cv_means = [0.25, 0.40, 0.55, 0.70, 0.85]
    data_points = []

    for mean_cv in cv_means:
        cv_range = (mean_cv - 0.07, mean_cv + 0.07)
        set_config({
            'CV_RANGE': cv_range,
            'W1': 0.40, 'W2': 0.25, 'W3': 0.35
        })

        res = run_batch_simulation(n_batches=30, label=f"CV={mean_cv}")

        data_points.append({
            'cv_mean': mean_cv,
            'baseline_cvr': np.mean(res['baseline_cvr']),
            'rosa_cvr': np.mean(res['rosa_cvr'])
        })
        print(f"  CV={mean_cv}: Baseline={np.mean(res['baseline_cvr']):.4f}, ROSA={np.mean(res['rosa_cvr']):.4f}")

    df = pd.DataFrame(data_points)
    df.to_csv("results_exp2_cv_sensitivity.csv", index=False)
    print(f"\nSaved to results_exp2_cv_sensitivity.csv")
    return df

def exp3_ablation_study():
    """实验3: 消融实验 - 高压环境"""
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
        'THETA': 0.80
    })
    
    # ROSA-Full
    print("\nRunning ROSA-Full (W3=0.35) under high pressure...")
    set_config({'W1': 0.40, 'W2': 0.25, 'W3': 0.35})
    res_full = run_batch_simulation(n_batches=30, label="ROSA-Full")
    
    # ROSA-NoRisk
    print("\nRunning ROSA-NoRisk (W3=0) under high pressure...")
    set_config({'W1': 0.40, 'W2': 0, 'W3': 0.35})
    res_norisk = run_batch_simulation(n_batches=30, label="ROSA-NoRisk")
    
    # Baseline作为参照
    cvr_full = np.mean(res_full['rosa_cvr'])
    cvr_norisk = np.mean(res_norisk['rosa_cvr'])
    cvr_baseline = np.mean(res_full['baseline_cvr'])
    
    print(f"\n[Ablation Results - High Pressure]")
    print(f"  Baseline:    CVR = {cvr_baseline:.4f}")
    print(f"  ROSA-Full:   CVR = {cvr_full:.4f}")
    print(f"  ROSA-NoRisk: CVR = {cvr_norisk:.4f}")
    
    if cvr_norisk > cvr_full and cvr_norisk > 0:
        reduction = (cvr_norisk - cvr_full) / cvr_norisk * 100
        print(f"  O3 reduces CVR by {reduction:.1f}%")
    else:
        print(f"  Warning: O3 effect not observed, may need more pressure")
    
    return {'baseline': cvr_baseline, 'full': cvr_full, 'norisk': cvr_norisk}

if __name__ == "__main__":
    print("="*60)
    print("ICDCS Paper Experiments")
    print("="*60)

    exp1_main_comparison()
    exp2_sensitivity_cv()
    exp3_ablation_study()

    print("\n" + "="*60)
    print("All experiments completed!")
    print("="*60)
