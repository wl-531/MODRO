"""RA-LNS: Risk-Aware Large Neighborhood Search (Simplified 3-Level)

字典序简化：5 层 → 3 层 + tie-break
  Phi(X) = (feas, -U_max, -O1)
  Tie-break: R_sum
"""
import time
import numpy as np
from typing import List, Tuple


# 数值常量
EPS_DIV = 1e-6    # 除零保护
TOL_FEAS = 1e-9   # 可行性判断
EPS_CMP = 1e-6    # 浮点比较容差


class RALNSSolution:
    """解的表示（支持增量更新）"""
    
    def __init__(self, servers, kappa):
        self.m = len(servers)
        self.kappa = kappa
        self.C = np.array([s.C for s in servers])
        self.L0 = np.array([s.L0 for s in servers])
        self.mu_sum = np.zeros(self.m)
        self.sigma_sq_sum = np.zeros(self.m)
        self.assignment = []
    
    @property
    def sigma_j(self):
        return np.sqrt(np.maximum(self.sigma_sq_sum, 0))
    
    @property
    def L_hat(self):
        return self.L0 + self.mu_sum + self.kappa * self.sigma_j
    
    @property
    def Gap(self):
        return self.C - self.L_hat
    
    @property
    def RD(self):
        """风险密度（用于热点定位）"""
        return self.sigma_j / np.maximum(self.Gap, EPS_DIV)
    
    @property
    def U_max(self):
        """Level-1: 最大利用率"""
        return float(np.max(self.L_hat / self.C))
    
    @property
    def O1(self):
        """Level-2: Makespan"""
        return float(np.max(self.L_hat))
    
    @property
    def R_sum(self):
        """Tie-break: 总风险密度"""
        return float(np.sum(self.RD))
    
    def is_feasible(self) -> bool:
        """Level-0: 可行性检查"""
        return bool(np.all(self.Gap >= -TOL_FEAS))
    
    def Phi(self) -> Tuple[int, float, float]:
        """3 层字典序向量：(feas, -U_max, -O1)"""
        return (
            1 if self.is_feasible() else 0,
            -self.U_max,
            -self.O1
        )
    
    def apply_move(self, task_idx, task, from_j, to_j):
        if from_j is not None and from_j != -1:
            self.mu_sum[from_j] -= task.mu
            self.sigma_sq_sum[from_j] -= task.sigma ** 2
        self.mu_sum[to_j] += task.mu
        self.sigma_sq_sum[to_j] += task.sigma ** 2
        self.assignment[task_idx] = to_j
    
    def rollback_move(self, task_idx, task, from_j, to_j):
        self.mu_sum[to_j] -= task.mu
        self.sigma_sq_sum[to_j] -= task.sigma ** 2
        if from_j is not None and from_j != -1:
            self.mu_sum[from_j] += task.mu
            self.sigma_sq_sum[from_j] += task.sigma ** 2
        self.assignment[task_idx] = from_j if from_j is not None else -1
    
    def copy(self):
        new_sol = RALNSSolution.__new__(RALNSSolution)
        new_sol.m = self.m
        new_sol.kappa = self.kappa
        new_sol.C = self.C.copy()
        new_sol.L0 = self.L0.copy()
        new_sol.mu_sum = self.mu_sum.copy()
        new_sol.sigma_sq_sum = self.sigma_sq_sum.copy()
        new_sol.assignment = self.assignment.copy()
        return new_sol


class RALNSSolver:
    """RA-LNS Solver (3-Level Lexicographic)"""
    
    def __init__(self, kappa, patience=15, destroy_k=3, t_max=0.01,
                 eps_div=1e-6, tol_feas=1e-9):
        self.kappa = kappa
        self.patience = patience
        self.destroy_k = destroy_k
        self.t_max = t_max
        self.eps_div = eps_div
        self.tol_feas = tol_feas
    
    def solve(self, tasks, servers) -> Tuple[List[int], int]:
        start = time.perf_counter()
        sol, fallback_count = self._risk_first_construction(tasks, servers)
        
        best = sol.copy() if sol.is_feasible() else None
        best_rsum = sol.R_sum if best else float("inf")
        stagnation = 0
        iteration = 0
        
        while time.perf_counter() - start < self.t_max:
            if stagnation < self.patience:
                improved = self._risk_hedging_move(sol, tasks)
            else:
                improved = self._risk_guided_lns(sol, tasks)
                stagnation = 0
            
            if improved:
                stagnation = 0
                if sol.is_feasible():
                    if best is None or self._lex_better(sol.Phi(), best.Phi(), sol.R_sum, best_rsum):
                        best = sol.copy()
                        best_rsum = sol.R_sum
            else:
                stagnation += 1
            
            iteration += 1
            if iteration > 1000:
                break
        
        result = best if best else sol
        assignment = result.assignment.copy()
        assert len(assignment) == len(tasks), "assignment 长度错误"
        assert all(a != -1 for a in assignment), "存在未分配任务"
        return assignment, fallback_count
    
    def _risk_first_construction(self, tasks, servers):
        """Phase 0: 按 delta_i 降序贪心分配"""
        sol = RALNSSolution(servers, self.kappa)
        fallback_count = 0
        n_tasks = len(tasks)
        
        deltas = [(i, tasks[i].mu + self.kappa * tasks[i].sigma) for i in range(n_tasks)]
        sorted_indices = [i for i, _ in sorted(deltas, key=lambda x: -x[1])]
        sol.assignment = [-1] * n_tasks
        
        for i in sorted_indices:
            task = tasks[i]
            new_sigma_sq = sol.sigma_sq_sum + task.sigma ** 2
            new_sigma = np.sqrt(np.maximum(new_sigma_sq, 0))
            new_mu = sol.mu_sum + task.mu
            new_L_hat = sol.L0 + new_mu + self.kappa * new_sigma
            new_Gap = sol.C - new_L_hat
            
            task_sigma = max(task.sigma, self.eps_div)
            scores = new_Gap / task_sigma
            
            best_j = None
            best_score = -np.inf
            for j in range(sol.m):
                if new_Gap[j] >= -self.tol_feas and scores[j] > best_score:
                    best_score = scores[j]
                    best_j = j
            
            if best_j is not None:
                sol.assignment[i] = best_j
                sol.mu_sum[best_j] += task.mu
                sol.sigma_sq_sum[best_j] += task.sigma ** 2
            else:
                fallback_count += 1
                j_min = int(np.argmin(sol.L_hat))
                sol.assignment[i] = j_min
                sol.mu_sum[j_min] += task.mu
                sol.sigma_sq_sum[j_min] += task.sigma ** 2
        
        return sol, fallback_count
    
    def _risk_hedging_move(self, sol, tasks) -> bool:
        """Stage-1A: Relocate / Swap"""
        j_hot = int(np.argmax(sol.RD))
        victims = [i for i, j in enumerate(sol.assignment) if j == j_hot]
        if not victims:
            return False
        
        victim_sigmas = [tasks[i].sigma for i in victims]
        victim_idx = victims[int(np.argmax(victim_sigmas))]
        victim_task = tasks[victim_idx]
        from_j = sol.assignment[victim_idx]
        
        best_move = None
        best_phi = sol.Phi()
        best_rsum = sol.R_sum
        
        for to_j in range(sol.m):
            if to_j == from_j:
                continue
            
            # Relocate
            sol.apply_move(victim_idx, victim_task, from_j, to_j)
            new_phi = sol.Phi()
            new_rsum = sol.R_sum
            if self._lex_better(new_phi, best_phi, new_rsum, best_rsum):
                best_phi = new_phi
                best_rsum = new_rsum
                best_move = ("relocate", victim_idx, victim_task, from_j, to_j)
            sol.rollback_move(victim_idx, victim_task, from_j, to_j)
            
            # Swap
            swap_cands = [i for i, j in enumerate(sol.assignment) if j == to_j]
            if swap_cands:
                swap_sigmas = [tasks[i].sigma for i in swap_cands]
                swap_idx = swap_cands[int(np.argmin(swap_sigmas))]
                swap_task = tasks[swap_idx]
                
                sol.apply_move(victim_idx, victim_task, from_j, to_j)
                sol.apply_move(swap_idx, swap_task, to_j, from_j)
                new_phi = sol.Phi()
                new_rsum = sol.R_sum
                if self._lex_better(new_phi, best_phi, new_rsum, best_rsum):
                    best_phi = new_phi
                    best_rsum = new_rsum
                    best_move = ("swap", victim_idx, victim_task, from_j, to_j, swap_idx, swap_task)
                sol.rollback_move(swap_idx, swap_task, to_j, from_j)
                sol.rollback_move(victim_idx, victim_task, from_j, to_j)
        
        if best_move:
            if best_move[0] == "relocate":
                _, vi, vt, fj, tj = best_move
                sol.apply_move(vi, vt, fj, tj)
            else:
                _, vi, vt, fj, tj, si, st = best_move
                sol.apply_move(vi, vt, fj, tj)
                sol.apply_move(si, st, tj, fj)
            return True
        return False
    
    def _risk_guided_lns(self, sol, tasks) -> bool:
        """Stage-1B: Destroy + Repair"""
        j_hot = int(np.argmax(sol.RD))
        victims = [i for i, j in enumerate(sol.assignment) if j == j_hot]
        if len(victims) < self.destroy_k:
            return False
        
        victim_sigmas = [(i, tasks[i].sigma) for i in victims]
        victim_sigmas.sort(key=lambda x: x[1], reverse=True)
        destroy_tasks = [i for i, _ in victim_sigmas[:self.destroy_k]]
        
        backup_sol = sol.copy()
        backup_rsum = sol.R_sum
        
        for i in destroy_tasks:
            task = tasks[i]
            j = sol.assignment[i]
            sol.mu_sum[j] -= task.mu
            sol.sigma_sq_sum[j] -= task.sigma ** 2
            sol.assignment[i] = -1
        
        repair_order = sorted(destroy_tasks, key=lambda i: tasks[i].sigma, reverse=True)
        for i in repair_order:
            task = tasks[i]
            new_sigma_sq = sol.sigma_sq_sum + task.sigma ** 2
            new_sigma = np.sqrt(np.maximum(new_sigma_sq, 0))
            new_mu = sol.mu_sum + task.mu
            new_L_hat = sol.L0 + new_mu + self.kappa * new_sigma
            new_Gap = sol.C - new_L_hat
            
            best_j = None
            best_delta_rd = np.inf
            for j in range(sol.m):
                if new_Gap[j] >= -self.tol_feas:
                    rd_before = sol.sigma_j[j] / max(sol.Gap[j], self.eps_div)
                    rd_after = new_sigma[j] / max(new_Gap[j], self.eps_div)
                    delta_rd = rd_after - rd_before
                    if delta_rd < best_delta_rd:
                        best_delta_rd = delta_rd
                        best_j = j
            
            if best_j is not None:
                sol.assignment[i] = best_j
                sol.mu_sum[best_j] += task.mu
                sol.sigma_sq_sum[best_j] += task.sigma ** 2
            else:
                j_min = int(np.argmin(sol.L_hat))
                sol.assignment[i] = j_min
                sol.mu_sum[j_min] += task.mu
                sol.sigma_sq_sum[j_min] += task.sigma ** 2
        
        if self._lex_better(sol.Phi(), backup_sol.Phi(), sol.R_sum, backup_rsum):
            return True
        else:
            sol.mu_sum = backup_sol.mu_sum.copy()
            sol.sigma_sq_sum = backup_sol.sigma_sq_sum.copy()
            sol.assignment = backup_sol.assignment.copy()
            return False
    
    def _lex_better(self, phi1, phi2, r_sum1, r_sum2) -> bool:
        """3 层字典序 + R_sum tie-break"""
        # Level-0: feas (严格比较)
        if phi1[0] != phi2[0]:
            return phi1[0] > phi2[0]
        
        # Level-1, Level-2: 浮点比较
        for v1, v2 in zip(phi1[1:], phi2[1:]):
            if v1 > v2 + EPS_CMP:
                return True
            if v1 < v2 - EPS_CMP:
                return False
        
        # Tie-break: R_sum 最小者胜
        if r_sum1 < r_sum2 - EPS_CMP:
            return True
        
        return False
