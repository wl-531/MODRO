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
from evaluation.server_monitor import ServerMonitor

def set_config(updates: dict):
    for key, val in updates.items():
        setattr(config, key, val)

# NEW: 添加VAG_LAMBDA到config（如果不存在）
if not hasattr(config, 'VAG_LAMBDA'):
    setattr(config, 'VAG_LAMBDA', 1.0)

def run_batch_simulation(n_batches=50, label="Experiment", enable_all_algos=True,
                         task_mode="coupled", enable_diagnostics=False):
    """运行批量仿真，支持4个算法：DG / VAG / NSGA2Mean / ROSA

    Args:
        n_batches: 批次数量
        label: 实验标签
        enable_all_algos: 是否启用所有算法（False时只运行DG和ROSA，兼容旧版）
        task_mode: 任务生成模式
            - "coupled": 原模式，σ = μ × CV（默认）
            - "bimodal": 双峰模式，μ-σ 解耦
        enable_diagnostics: 是否启用服务器状态监控（诊断用）

    Returns:
        results: {algo_name: {'cvr': [...], 'L0': [...]}}
    """
    print(f"\n>>> Running {label} (n={n_batches}, task_mode={task_mode})...")
    np.random.seed(42)

    # 初始化监控器（如果启用）
    monitor = ServerMonitor() if enable_diagnostics else None

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
        if monitor:
            monitor.record_batch(i, "dg", assign_dg, tasks, servers_dg)
        update_server_state(servers_dg, assign_dg, tasks, config.DECISION_INTERVAL)

        # ROSA算法
        assign_rosa = rosa.solve_batch(tasks, servers_rosa)
        cvr_rosa = monte_carlo_cvr(assign_rosa, tasks, servers_rosa, config.MC_SAMPLES)
        L0_rosa = sum(s.L0 for s in servers_rosa)
        if monitor:
            monitor.record_batch(i, "rosa", assign_rosa, tasks, servers_rosa)
        update_server_state(servers_rosa, assign_rosa, tasks, config.DECISION_INTERVAL)

        # NEW: VAG算法
        if enable_all_algos:
            assign_vag = variance_aware_greedy(tasks, servers_vag, lambda_=config.VAG_LAMBDA)
            cvr_vag = monte_carlo_cvr(assign_vag, tasks, servers_vag, config.MC_SAMPLES)
            L0_vag = sum(s.L0 for s in servers_vag)
            if monitor:
                monitor.record_batch(i, "vag", assign_vag, tasks, servers_vag)
            update_server_state(servers_vag, assign_vag, tasks, config.DECISION_INTERVAL)

        # NEW: NSGA2Mean算法
        if enable_all_algos:
            assign_nsga = nsga.solve_batch(tasks, servers_nsga)
            cvr_nsga = monte_carlo_cvr(assign_nsga, tasks, servers_nsga, config.MC_SAMPLES)
            L0_nsga = sum(s.L0 for s in servers_nsga)
            if monitor:
                monitor.record_batch(i, "nsga", assign_nsga, tasks, servers_nsga)
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

    # 导出诊断数据（如果启用）
    if monitor:
        csv_filename = f"diagnostics_{label.lower().replace(' ', '_')}.csv"
        monitor.export_csv(csv_filename)
        monitor.print_summary()

    return results

def exp1_main_comparison():
    """实验1: 中等压力，多类型任务（multiclass），对比 DG / NSGA-Mean / ROSA 时序表现"""
    print("\n" + "="*60)
    print("EXP 1: Moderate Pressure, Multiclass Tasks (DG / NSGA-Mean / ROSA)")
    print("="*60)

    # 中等压力配置
    set_config({
        'N_SERVERS': 5,
        'BATCH_SIZE': 80,
        'DECISION_INTERVAL': 8,
        'ALPHA': 0.15,
        'KAPPA': 2.38,
        'THETA': 1.0,
        'W1': 0.40, 'W2': 0.25, 'W3': 0.35,
        'VAG_LAMBDA': 1.0
    })

    n_batches = 50
    res = run_batch_simulation(
        n_batches=n_batches,
        label="Exp1_Moderate_Bimodal",
        enable_all_algos=True,
        task_mode="bimodal"
    )

    # 仅关注 DG / NSGA / ROSA 的概览
    print(f"\n[Summary - Exp1]")
    for algo_name in ["dg", "nsga", "rosa"]:
        cvr_mean = np.mean(res[algo_name]["cvr"])
        cvr_std = np.std(res[algo_name]["cvr"])
        L0_mean = np.mean(res[algo_name]["L0"])
        print(f"{algo_name.upper():8s}: CVR = {cvr_mean:.4f} ± {cvr_std:.4f}, Avg L0 = {L0_mean:.1f}")

    # 构造时序 CSV（每个 batch 一行）
    rows = []
    for algo_name in ["dg", "nsga", "rosa"]:
        cvr_series = res[algo_name]["cvr"]
        L0_series = res[algo_name]["L0"]
        for t in range(n_batches):
            rows.append({
                "exp": "exp1_moderate",
                "algo": algo_name,
                "batch_idx": t + 1,
                "cvr": float(cvr_series[t]),
                "L0": float(L0_series[t]),
            })
    df = pd.DataFrame(rows)
    df.to_csv("results_exp1_timeseries.csv", index=False)
    print("Saved per-batch results to results_exp1_timeseries.csv")

    return res

def exp2_sensitivity_cv():
    """实验2: 高压力双峰场景，多任务批次，对比 DG / NSGA-Mean / ROSA 时序表现"""
    print("\n" + "="*60)
    print("EXP 2: High-Pressure Bimodal Tasks (DG / NSGA-Mean / ROSA)")
    print("="*60)

    # 高压配置：双峰任务分布，批次任务量大
    set_config({
        'N_SERVERS': 10,
        'BATCH_SIZE': 800,
        'DECISION_INTERVAL': 53.0,
        'ALPHA': 0.15,
        'KAPPA': 2.38,
        'THETA': 1.0,
        'W1': 0.40, 'W2': 0.25, 'W3': 0.35,
        'VAG_LAMBDA': 1.0
    })

    n_batches = 50
    res = run_batch_simulation(
        n_batches=n_batches,
        label="Exp2_HighPressure_Bimodal",
        enable_all_algos=True,
        task_mode="bimodal"
    )

    # 仅关注 DG / NSGA / ROSA 的概览
    print(f"\n[Summary - Exp2]")
    for algo_name in ["dg", "nsga", "rosa"]:
        cvr_mean = np.mean(res[algo_name]["cvr"])
        cvr_std = np.std(res[algo_name]["cvr"])
        L0_mean = np.mean(res[algo_name]["L0"])
        print(f"{algo_name.upper():8s}: CVR = {cvr_mean:.4f} ± {cvr_std:.4f}, Avg L0 = {L0_mean:.1f}")

    # 构造时序 CSV
    rows = []
    for algo_name in ["dg", "nsga", "rosa"]:
        cvr_series = res[algo_name]["cvr"]
        L0_series = res[algo_name]["L0"]
        for t in range(n_batches):
            rows.append({
                "exp": "exp2_high_pressure",
                "algo": algo_name,
                "batch_idx": t + 1,
                "cvr": float(cvr_series[t]),
                "L0": float(L0_series[t]),
            })
    df = pd.DataFrame(rows)
    df.to_csv("results_exp2_timeseries.csv", index=False)
    print("Saved per-batch results to results_exp2_timeseries.csv")

    return res

def exp3_ablation_study():
    """
    实验3: ROSA 消融实验（风险目标 O3）- 高压双峰多任务场景

    场景与 Exp2 一致：高压力、多任务、bimodal 任务分布（μ-σ 解耦）。
    对比:
      - DG (risk-neutral greedy)
      - NSGA-Mean (risk-neutral 多目标)
      - ROSA-Full (W3 > 0, 带边际不确定性风险目标)
      - ROSA-NoRisk (W3 = 0, 去掉 O3 但保留鲁棒初始化和去风险变异)
    """
    print("\n" + "="*60)
    print("EXP 3: ROSA Ablation (Risk Objective O3) - High Pressure Bimodal")
    print("="*60)

    # 1) 设置高压双峰配置（和 Exp2 高压场景保持一致）
    set_config({
        "N_SERVERS": 10,
        "BATCH_SIZE": 800,
        "DECISION_INTERVAL": 53.0,
        "ALPHA": 0.15,
        "KAPPA": 2.38,
        "THETA": 1.0,
        # ROSA 目标权重：完整版本
        "W1": 0.40,
        "W2": 0.25,
        "W3": 0.35,
    })

    n_batches = 50

    # 2) 先运行 DG / NSGA / ROSA-Full（高压双峰多任务）
    print("\nRunning DG / NSGA-Mean / ROSA-Full under high-pressure bimodal scenario...")
    res_full = run_batch_simulation(
        n_batches=n_batches,
        label="Exp3_Ablation_Full",
        enable_all_algos=True,      # DG + VAG + NSGA + ROSA 全部跑，下面只用 DG/NSGA/ROSA
        task_mode="bimodal",
    )

    # 3) 在同一场景下运行 ROSA-NoRisk（W3 = 0，仅去掉风险目标 O3）
    print("\nRunning ROSA-NoRisk (W3=0, no risk objective) under the same scenario...")
    set_config({"W3": 0.0})
    res_norisk = run_batch_simulation(
        n_batches=n_batches,
        label="Exp3_Ablation_ROSA_NoRisk",
        enable_all_algos=False,     # 这里只需要 DG + ROSA，VAG/NSGA 不必重复
        task_mode="bimodal",
    )

    # 4) 汇总 CVR 统计信息（均值 / 标准差 / 最大值）
    def summarize_cvr(series):
        series = np.asarray(series, dtype=float)
        return float(np.mean(series)), float(np.std(series)), float(np.max(series))

    cvr_dg_mean, cvr_dg_std, cvr_dg_max = summarize_cvr(res_full["dg"]["cvr"])
    cvr_nsga_mean, cvr_nsga_std, cvr_nsga_max = summarize_cvr(res_full["nsga"]["cvr"])
    cvr_rosa_full_mean, cvr_rosa_full_std, cvr_rosa_full_max = summarize_cvr(res_full["rosa"]["cvr"])
    cvr_rosa_norisk_mean, cvr_rosa_norisk_std, cvr_rosa_norisk_max = summarize_cvr(res_norisk["rosa"]["cvr"])

    print("\n[Ablation Summary]  CVR mean ± std  (max)")
    print(f"  DG          : {cvr_dg_mean:.4f} ± {cvr_dg_std:.4f} (max={cvr_dg_max:.4f})")
    print(f"  NSGA-Mean   : {cvr_nsga_mean:.4f} ± {cvr_nsga_std:.4f} (max={cvr_nsga_max:.4f})")
    print(f"  ROSA-Full   : {cvr_rosa_full_mean:.4f} ± {cvr_rosa_full_std:.4f} (max={cvr_rosa_full_max:.4f})")
    print(f"  ROSA-NoRisk : {cvr_rosa_norisk_mean:.4f} ± {cvr_rosa_norisk_std:.4f} (max={cvr_rosa_norisk_max:.4f})")

    # 5) 构造逐 batch 的时序 CSV：方便后续画 ROSA-Full vs ROSA-NoRisk 的曲线
    rows = []

    # (a) Full 实验：DG / NSGA / ROSA-Full
    for algo_name in ["dg", "nsga", "rosa"]:
        cvr_series = res_full[algo_name]["cvr"]
        L0_series = res_full[algo_name]["L0"]
        for t in range(n_batches):
            rows.append({
                "exp": "exp3_ablation",
                "algo": algo_name,
                # DG / NSGA 视作 baseline，ROSA 视作 full
                "variant": "baseline" if algo_name in ["dg", "nsga"] else "full",
                "batch_idx": int(t + 1),
                "cvr": float(cvr_series[t]),
                "L0": float(L0_series[t]),
            })

    # (b) ROSA-NoRisk 实验
    cvr_series = res_norisk["rosa"]["cvr"]
    L0_series = res_norisk["rosa"]["L0"]
    for t in range(n_batches):
        rows.append({
            "exp": "exp3_ablation",
            "algo": "rosa",
            "variant": "norisk",     # 去掉风险目标 O3
            "batch_idx": int(t + 1),
            "cvr": float(cvr_series[t]),
            "L0": float(L0_series[t]),
        })

    df = pd.DataFrame(rows)
    df.to_csv("results_exp3_timeseries.csv", index=False)
    print("\nSaved per-batch ablation results to results_exp3_timeseries.csv")

    # 6) 恢复 W3 的默认值，避免影响后续其它实验
    set_config({"W3": 0.35})

    return {
        "dg_mean_cvr": cvr_dg_mean,
        "nsga_mean_cvr": cvr_nsga_mean,
        "rosa_full_mean_cvr": cvr_rosa_full_mean,
        "rosa_norisk_mean_cvr": cvr_rosa_norisk_mean,
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
    print("Paper Experiments: Exp1 + Exp2")
    print("="*60)

    exp1_main_comparison()
    exp2_sensitivity_cv()
    exp3_ablation_study()

    print("\n" + "="*60)
    print("All experiments completed!")
    print("="*60)
