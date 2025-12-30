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
    np.random.seed(42)
    
    ra_lns = RALNSSolver(
        kappa=KAPPA, patience=RA_LNS_PATIENCE, destroy_k=RA_LNS_DESTROY_K,
        t_max=RA_LNS_T_MAX, eps_div=RA_LNS_EPS_DIV, tol_feas=RA_LNS_TOL_FEAS
    )
    
    servers_init = generate_servers(N_SERVERS, decision_interval=DECISION_INTERVAL)
    sample_tasks = generate_tasks(BATCH_SIZE, MU_RANGE, CV_RANGE, mode=task_mode)
    mu_vals = np.array([t.mu for t in sample_tasks])
    sigma_vals = np.array([t.sigma for t in sample_tasks])
    mu_avg = float(np.mean(mu_vals))
    cv_avg = float(np.mean(sigma_vals / np.maximum(mu_vals, 1e-6)))
    validation = validate_system_params(servers_init, BATCH_SIZE, mu_avg, cv_avg, KAPPA, 1.0)
    
    print("===== System Validation =====")
    print(f"rho_expected: {validation['rho_expected']:.3f}, rho_robust: {validation['rho_robust']:.3f}")
    print(f"Feasible: {'OK' if validation['feasible'] else 'FAIL'}\n")
    
    servers_dg = deepcopy(servers_init)
    servers_vag = deepcopy(servers_init)
    servers_kappa = deepcopy(servers_init)
    servers_ralns = deepcopy(servers_init)
    
    def init_results():
        return {'cvr': [], 'O1': [], 'time_ms': [], 'residual': []}
    
    results_dg = init_results()
    results_vag = init_results()
    results_kappa = init_results()
    results_ralns = init_results()
    results_ralns['fallback'] = []
    
    print("===== Experiment Start =====")
    for batch_idx in range(n_batches):
        tasks = generate_tasks(BATCH_SIZE, MU_RANGE, CV_RANGE, mode=task_mode)
        
        # DG
        t0 = time.perf_counter()
        assign_dg = deterministic_greedy(tasks, servers_dg)
        time_dg = (time.perf_counter() - t0) * 1000
        cvr_dg = monte_carlo_cvr(assign_dg, tasks, servers_dg, MC_SAMPLES)
        metrics_dg = compute_metrics_unified(assign_dg, tasks, servers_dg, KAPPA)
        results_dg['cvr'].append(cvr_dg)
        results_dg['O1'].append(metrics_dg['O1'])
        results_dg['time_ms'].append(time_dg)
        
        # VAG
        t0 = time.perf_counter()
        assign_vag = variance_aware_greedy(tasks, servers_vag, lambda_=1.0)
        time_vag = (time.perf_counter() - t0) * 1000
        cvr_vag = monte_carlo_cvr(assign_vag, tasks, servers_vag, MC_SAMPLES)
        metrics_vag = compute_metrics_unified(assign_vag, tasks, servers_vag, KAPPA)
        results_vag['cvr'].append(cvr_vag)
        results_vag['O1'].append(metrics_vag['O1'])
        results_vag['time_ms'].append(time_vag)
        
        # kappa-Greedy
        t0 = time.perf_counter()
        assign_kappa = kappa_greedy(tasks, servers_kappa, kappa=KAPPA)
        time_kappa = (time.perf_counter() - t0) * 1000
        cvr_kappa = monte_carlo_cvr(assign_kappa, tasks, servers_kappa, MC_SAMPLES)
        metrics_kappa = compute_metrics_unified(assign_kappa, tasks, servers_kappa, KAPPA)
        results_kappa['cvr'].append(cvr_kappa)
        results_kappa['O1'].append(metrics_kappa['O1'])
        results_kappa['time_ms'].append(time_kappa)
        
        # RA-LNS
        t0 = time.perf_counter()
        assign_ralns, fb = ra_lns.solve(tasks, servers_ralns)
        time_ralns = (time.perf_counter() - t0) * 1000
        cvr_ralns = monte_carlo_cvr(assign_ralns, tasks, servers_ralns, MC_SAMPLES)
        metrics_ralns = compute_metrics_unified(assign_ralns, tasks, servers_ralns, KAPPA)
        results_ralns['cvr'].append(cvr_ralns)
        results_ralns['O1'].append(metrics_ralns['O1'])
        results_ralns['time_ms'].append(time_ralns)
        results_ralns['fallback'].append(fb)
        
        # Update states
        update_server_state(servers_dg, assign_dg, tasks, DECISION_INTERVAL)
        update_server_state(servers_vag, assign_vag, tasks, DECISION_INTERVAL)
        update_server_state(servers_kappa, assign_kappa, tasks, DECISION_INTERVAL)
        update_server_state(servers_ralns, assign_ralns, tasks, DECISION_INTERVAL)
        
        results_dg['residual'].append(sum(s.L0 for s in servers_dg))
        results_vag['residual'].append(sum(s.L0 for s in servers_vag))
        results_kappa['residual'].append(sum(s.L0 for s in servers_kappa))
        results_ralns['residual'].append(sum(s.L0 for s in servers_ralns))
        
        if verbose:
            print(f"Batch {batch_idx+1}/{n_batches}: "
                  f"DG={cvr_dg:.4f} VAG={cvr_vag:.4f} kappa={cvr_kappa:.4f} RA-LNS={cvr_ralns:.4f} "
                  f"[{time_dg:.1f}/{time_vag:.1f}/{time_kappa:.1f}/{time_ralns:.1f}ms]")
    
    # Results
    print("\n===== Results =====")
    def pr(name, r):
        print(f"[{name}]")
        print(f"  CVR: {np.mean(r['cvr']):.4f} +/- {np.std(r['cvr']):.4f}")
        print(f"  O1:  {np.mean(r['O1']):.1f}")
        print(f"  Time: {np.mean(r['time_ms']):.2f}ms (p50={np.percentile(r['time_ms'], 50):.2f}, p99={np.percentile(r['time_ms'], 99):.2f})")
    
    pr("DG", results_dg)
    pr("VAG", results_vag)
    pr("kappa-Greedy", results_kappa)
    pr("RA-LNS", results_ralns)
    print(f"  Fallback: {sum(results_ralns['fallback'])}")
    
    # Comparison
    base = np.mean(results_dg['cvr'])
    kappa_cvr = np.mean(results_kappa['cvr'])
    ralns_cvr = np.mean(results_ralns['cvr'])
    print(f"\n[Comparison]")
    print(f"  kappa vs DG: {(base-kappa_cvr)/max(base,1e-6)*100:+.1f}%")
    print(f"  RA-LNS vs DG: {(base-ralns_cvr)/max(base,1e-6)*100:+.1f}%")
    print(f"  RA-LNS vs kappa: {(kappa_cvr-ralns_cvr)/max(kappa_cvr,1e-6)*100:+.1f}%")
    
    status = "OK" if ralns_cvr < ALPHA else "WARN"
    print(f"  RA-LNS CVR={ralns_cvr:.4f} vs alpha={ALPHA} [{status}]")
    
    return {'dg': results_dg, 'vag': results_vag, 'kappa': results_kappa, 'ralns': results_ralns}


if __name__ == '__main__':
    run_experiment(n_batches=10, verbose=True, task_mode="bimodal")
