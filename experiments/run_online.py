"""微批处理在线实验 - 基线算法 vs RA-LNS 对比"""
import numpy as np
import time
import sys
import os
from copy import deepcopy
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import *
from models.task import Task
from models.server import Server
from solvers.baselines import deterministic_greedy, variance_aware_greedy
from solvers.kappa_greedy import kappa_greedy
from solvers.ra_lns import RALNSSolver
from evaluation.metrics_unified import compute_metrics_unified
from evaluation.monte_carlo import monte_carlo_cvr
from data.generator import generate_tasks, generate_servers, validate_system_params


def update_server_state(servers, assignment, tasks, processing_time):
    """更新服务器已有负载（采样实际工作量）"""
    n_servers = len(servers)
    actual_load = np.zeros(n_servers)
    
    for i, j in enumerate(assignment):
        w_actual = np.random.normal(tasks[i].mu, tasks[i].sigma)
        w_actual = max(0.0, w_actual)
        actual_load[j] += w_actual
    
    for j in range(n_servers):
        processed = servers[j].f * processing_time
        servers[j].L0 = max(0.0, servers[j].L0 + actual_load[j] - processed)


def run_experiment(n_batches=10, verbose=True, task_mode="coupled"):
    """运行微批处理在线实验 - 基线算法 vs RA-LNS 对比"""
    
    # 固定随机种子确保可复现
    np.random.seed(42)
    
    # 初始化 RA-LNS solver
    ra_lns = RALNSSolver(
        kappa=KAPPA,
        patience=RA_LNS_PATIENCE,
        destroy_k=RA_LNS_DESTROY_K,
        t_max=RA_LNS_T_MAX,
        eps_div=RA_LNS_EPS_DIV,
        tol_feas=RA_LNS_TOL_FEAS
    )
    
    # 生成初始服务器
    servers_init = generate_servers(N_SERVERS, decision_interval=DECISION_INTERVAL)
    
    # 验证系统参数
    sample_tasks = generate_tasks(BATCH_SIZE, MU_RANGE, CV_RANGE, mode=task_mode)
    mu_vals = np.array([t.mu for t in sample_tasks])
    sigma_vals = np.array([t.sigma for t in sample_tasks])
    mu_avg = float(np.mean(mu_vals))
    cv_avg = float(np.mean(sigma_vals / np.maximum(mu_vals, 1e-6)))
    validation = validate_system_params(servers_init, BATCH_SIZE, mu_avg, cv_avg, KAPPA, 1.0)
    
    print("===== 系统参数验证 =====")
    print(f"总容量: {validation['total_capacity']:.0f}")
    print(f"期望负载: {validation['expected_load']:.0f}")
    print(f"鲁棒负载(分散): {validation['robust_load']:.0f}")
    print(f"期望负载率 rho: {validation['rho_expected']:.3f}")
    print(f"鲁棒负载率 rho_robust: {validation['rho_robust']:.3f}")
    print(f"系统可行性: {'[OK] 可行' if validation['feasible'] else '[FAIL] 不可行'}")
    print()
    
    # 创建独立环境
    servers_dg = deepcopy(servers_init)
    servers_vag = deepcopy(servers_init)
    servers_kappa = deepcopy(servers_init)
    servers_ralns = deepcopy(servers_init)
    
    # 初始化结果记录
    def init_results():
        return {
            'cvr': [],
            'U_max': [],
            'O1': [],
            'R_sum': [],
            'O2': [],
            'residual': [],
            'time_ms': []
        }
    
    results_dg = init_results()
    results_vag = init_results()
    results_kappa = init_results()
    results_ralns = init_results()
    results_ralns['fallback_count'] = []
    
    print("===== 基线算法 vs RA-LNS 对比实验 =====\n")
    
    for batch_idx in range(n_batches):
        # 生成相同的任务批次
        tasks = generate_tasks(BATCH_SIZE, MU_RANGE, CV_RANGE, mode=task_mode)
        
        # ═══════ DG ═══════
        t0 = time.perf_counter()
        assign_dg = deterministic_greedy(tasks, servers_dg)
        time_dg = (time.perf_counter() - t0) * 1000
        
        cvr_dg = monte_carlo_cvr(assign_dg, tasks, servers_dg, MC_SAMPLES)
        metrics_dg = compute_metrics_unified(assign_dg, tasks, servers_dg, KAPPA, RA_LNS_EPS_DIV, RA_LNS_TOL_FEAS)
        
        results_dg['cvr'].append(cvr_dg)
        results_dg['U_max'].append(metrics_dg['U_max'])
        results_dg['O1'].append(metrics_dg['O1'])
        results_dg['R_sum'].append(metrics_dg['R_sum'])
        results_dg['O2'].append(metrics_dg['O2'])
        results_dg['time_ms'].append(time_dg)
        
        # ═══════ VAG ═══════
        t0 = time.perf_counter()
        assign_vag = variance_aware_greedy(tasks, servers_vag, lambda_=1.0)
        time_vag = (time.perf_counter() - t0) * 1000
        
        cvr_vag = monte_carlo_cvr(assign_vag, tasks, servers_vag, MC_SAMPLES)
        metrics_vag = compute_metrics_unified(assign_vag, tasks, servers_vag, KAPPA, RA_LNS_EPS_DIV, RA_LNS_TOL_FEAS)
        
        results_vag['cvr'].append(cvr_vag)
        results_vag['U_max'].append(metrics_vag['U_max'])
        results_vag['O1'].append(metrics_vag['O1'])
        results_vag['R_sum'].append(metrics_vag['R_sum'])
        results_vag['O2'].append(metrics_vag['O2'])
        results_vag['time_ms'].append(time_vag)
        
        # ═══════ κ-Greedy ═══════
        t0 = time.perf_counter()
        assign_kappa = kappa_greedy(tasks, servers_kappa, kappa=KAPPA)
        time_kappa = (time.perf_counter() - t0) * 1000
        
        cvr_kappa = monte_carlo_cvr(assign_kappa, tasks, servers_kappa, MC_SAMPLES)
        metrics_kappa = compute_metrics_unified(assign_kappa, tasks, servers_kappa, KAPPA, RA_LNS_EPS_DIV, RA_LNS_TOL_FEAS)
        
        results_kappa['cvr'].append(cvr_kappa)
        results_kappa['U_max'].append(metrics_kappa['U_max'])
        results_kappa['O1'].append(metrics_kappa['O1'])
        results_kappa['R_sum'].append(metrics_kappa['R_sum'])
        results_kappa['O2'].append(metrics_kappa['O2'])
        results_kappa['time_ms'].append(time_kappa)
        
        # ═══════ RA-LNS ═══════
        t0 = time.perf_counter()
        assign_ralns, fb_count = ra_lns.solve(tasks, servers_ralns)
        time_ralns = (time.perf_counter() - t0) * 1000
        
        cvr_ralns = monte_carlo_cvr(assign_ralns, tasks, servers_ralns, MC_SAMPLES)
        metrics_ralns = compute_metrics_unified(assign_ralns, tasks, servers_ralns, KAPPA, RA_LNS_EPS_DIV, RA_LNS_TOL_FEAS)
        
        results_ralns['cvr'].append(cvr_ralns)
        results_ralns['U_max'].append(metrics_ralns['U_max'])
        results_ralns['O1'].append(metrics_ralns['O1'])
        results_ralns['R_sum'].append(metrics_ralns['R_sum'])
        results_ralns['O2'].append(metrics_ralns['O2'])
        results_ralns['time_ms'].append(time_ralns)
        results_ralns['fallback_count'].append(fb_count)
        
        # 更新服务器状态
        update_server_state(servers_dg, assign_dg, tasks, DECISION_INTERVAL)
        update_server_state(servers_vag, assign_vag, tasks, DECISION_INTERVAL)
        update_server_state(servers_kappa, assign_kappa, tasks, DECISION_INTERVAL)
        update_server_state(servers_ralns, assign_ralns, tasks, DECISION_INTERVAL)
        
        residual_dg = sum(s.L0 for s in servers_dg)
        residual_vag = sum(s.L0 for s in servers_vag)
        residual_kappa = sum(s.L0 for s in servers_kappa)
        residual_ralns = sum(s.L0 for s in servers_ralns)
        
        results_dg['residual'].append(residual_dg)
        results_vag['residual'].append(residual_vag)
        results_kappa['residual'].append(residual_kappa)
        results_ralns['residual'].append(residual_ralns)
        
        if verbose:
            print(f"Batch {batch_idx + 1}/{n_batches}: "
                  f"DG_CVR={cvr_dg:.4f} | "
                  f"VAG_CVR={cvr_vag:.4f} | "
                  f"κ_CVR={cvr_kappa:.4f} | "
                  f"RA-LNS_CVR={cvr_ralns:.4f} "
                  f"[Time: DG={time_dg:.2f}ms, VAG={time_vag:.2f}ms, "
                  f"κ={time_kappa:.2f}ms, RA-LNS={time_ralns:.2f}ms]")
    
    # 汇总结果
    print("\n===== 实验结果汇总 =====")
    
    def print_results(name, results):
        print(f"\n[{name}]")
        print(f"  CVR:    {np.mean(results['cvr']):.4f} ± {np.std(results['cvr']):.4f}")
        print(f"  U_max:  {np.mean(results['U_max']):.4f}")
        print(f"  O1:     {np.mean(results['O1']):.2f}")
        print(f"  R_sum:  {np.mean(results['R_sum']):.4f}")
        print(f"  O2:     {np.mean(results['O2']):.2f}")
        print(f"  Residual: {np.mean(results['residual']):.1f}")
        print(f"  Time:   {np.mean(results['time_ms']):.2f}ms "
              f"(p50={np.percentile(results['time_ms'], 50):.2f}, "
              f"p99={np.percentile(results['time_ms'], 99):.2f})")
    
    print_results("DG - Deterministic Greedy", results_dg)
    print_results("VAG - Variance-Aware Greedy", results_vag)
    print_results("κ-Greedy - Robust Greedy", results_kappa)
    print_results("RA-LNS - Risk-Aware LNS", results_ralns)
    print(f"  Fallback total: {sum(results_ralns['fallback_count'])}")
    
    # 对比结果
    print(f"\n[对比结果]")
    baseline_cvr = np.mean(results_dg['cvr'])
    vag_cvr = np.mean(results_vag['cvr'])
    kappa_cvr = np.mean(results_kappa['cvr'])
    ralns_cvr = np.mean(results_ralns['cvr'])
    
    print(f"  VAG vs DG: {((baseline_cvr - vag_cvr) / max(baseline_cvr, 1e-6)) * 100:+.1f}%")
    print(f"  κ-Greedy vs DG: {((baseline_cvr - kappa_cvr) / max(baseline_cvr, 1e-6)) * 100:+.1f}%")
    print(f"  RA-LNS vs DG: {((baseline_cvr - ralns_cvr) / max(baseline_cvr, 1e-6)) * 100:+.1f}%")
    print(f"  RA-LNS vs κ-Greedy: {((kappa_cvr - ralns_cvr) / max(kappa_cvr, 1e-6)) * 100:+.1f}%")
    
    if ralns_cvr < ALPHA:
        print(f"  RA-LNS性能: [成功] CVR ({ralns_cvr:.4f}) < α ({ALPHA})")
    else:
        print(f"  RA-LNS性能: [警告] CVR ({ralns_cvr:.4f}) >= α ({ALPHA})")
    
    return {
        'dg': results_dg,
        'vag': results_vag,
        'kappa': results_kappa,
        'ralns': results_ralns
    }


if __name__ == '__main__':
    results = run_experiment(n_batches=10, verbose=True, task_mode="bimodal")
