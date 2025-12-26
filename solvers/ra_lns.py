"""RA-LNS: Risk-Aware Large Neighborhood Search"""
import time
import numpy as np
from typing import List, Tuple


class RALNSSolution:
    """解的表示（支持增量更新）"""
    
    def __init__(self, servers, kappa, eps_div=1e-6, tol_feas=1e-9):
        self.m = len(servers)
        self.kappa = kappa
        self.eps_div = eps_div
        self.tol_feas = tol_feas
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
        return self.sigma_j / np.maximum(self.Gap, self.eps_div)
    
    @property
    def U_max(self):
        return np.max(self.L_hat / self.C)
    
    @property
    def O1(self):
        return np.max(self.L_hat)
    
    @property
    def R_sum(self):
        return np.sum(self.RD)
    
    @property
    def O2(self):
        L_bar = np.mean(self.L_hat)
        return np.sum((self.L_hat - L_bar) ** 2)
    
    def Phi(self):
        return (self.O1, self.R_sum, self.O2)
    
    def is_feasible(self):
        return np.all(self.Gap >= -self.tol_feas)
    
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
        new_sol.eps_div = self.eps_div
        new_sol.tol_feas = self.tol_feas
        new_sol.C = self.C.copy()
        new_sol.L0 = self.L0.copy()
        new_sol.mu_sum = self.mu_sum.copy()
        new_sol.sigma_sq_sum = self.sigma_sq_sum.copy()
        new_sol.assignment = self.assignment.copy()
        return new_sol


class RALNSSolver:
    """RA-LNS Solver"""
    
    def __init__(self, kappa, patience=15, destroy_k=3, t_max=0.01,
                 eps_div=1e-6, tol_feas=1e-9):
        self.kappa = kappa
        self.patience = patience
        self.destroy_k = destroy_k
        self.t_max = t_max
        self.eps_div = eps_div
        self.tol_feas = tol_feas
    
    def solve(self, tasks, servers):
        start = time.perf_counter()
        sol, fallback_count = self._risk_first_construction(tasks, servers)
        best = sol.copy() if sol.is_feasible() else None
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
                    if best is None or self._lex_better(sol.Phi(), best.Phi()):
                        best = sol.copy()
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
        sol = RALNSSolution(servers, self.kappa, self.eps_div, self.tol_feas)
        fallback_count = 0
        
        for i, task in enumerate(tasks):
            Gap = sol.C - sol.L_hat
            new_mu = sol.mu_sum + task.mu
            new_sigma_sq = sol.sigma_sq_sum + task.sigma ** 2
            new_sigma = np.sqrt(np.maximum(new_sigma_sq, 0))
            new_L_hat = sol.L0 + new_mu + self.kappa * new_sigma
            new_Gap = sol.C - new_L_hat
            task_sigma = max(task.sigma, self.eps_div)
            scores = new_Gap / task_sigma
            j_best = int(np.argmax(scores))
            
            if new_Gap[j_best] < -self.tol_feas:
                j_best = int(np.argmax(Gap))
                fallback_count += 1
            
            sol.assignment.append(j_best)
            sol.mu_sum[j_best] += task.mu
            sol.sigma_sq_sum[j_best] += task.sigma ** 2
        
        return sol, fallback_count
    
    def _risk_hedging_move(self, sol, tasks):
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
        
        for to_j in range(sol.m):
            if to_j == from_j:
                continue
            sol.apply_move(victim_idx, victim_task, from_j, to_j)
            new_phi = sol.Phi()
            if sol.is_feasible() and self._lex_better(new_phi, best_phi):
                best_phi = new_phi
                best_move = (victim_idx, victim_task, from_j, to_j)
            sol.rollback_move(victim_idx, victim_task, from_j, to_j)
        
        if best_move:
            victim_idx, victim_task, from_j, to_j = best_move
            sol.apply_move(victim_idx, victim_task, from_j, to_j)
            return True
        return False
    
    def _risk_guided_lns(self, sol, tasks):
        j_hot = int(np.argmax(sol.RD))
        victims = [i for i, j in enumerate(sol.assignment) if j == j_hot]
        if len(victims) < self.destroy_k:
            return False
        
        victim_sigmas = [(i, tasks[i].sigma) for i in victims]
        victim_sigmas.sort(key=lambda x: x[1], reverse=True)
        destroy_tasks = [i for i, _ in victim_sigmas[:self.destroy_k]]
        backup_sol = sol.copy()
        
        for i in destroy_tasks:
            task = tasks[i]
            j = sol.assignment[i]
            sol.mu_sum[j] -= task.mu
            sol.sigma_sq_sum[j] -= task.sigma ** 2
            sol.assignment[i] = -1
        
        for i in destroy_tasks:
            task = tasks[i]
            new_mu = sol.mu_sum + task.mu
            new_sigma_sq = sol.sigma_sq_sum + task.sigma ** 2
            new_sigma = np.sqrt(np.maximum(new_sigma_sq, 0))
            new_L_hat = sol.L0 + new_mu + self.kappa * new_sigma
            new_Gap = sol.C - new_L_hat
            task_sigma = max(task.sigma, self.eps_div)
            scores = new_Gap / task_sigma
            j_best = int(np.argmax(scores))
            sol.assignment[i] = j_best
            sol.mu_sum[j_best] += task.mu
            sol.sigma_sq_sum[j_best] += task.sigma ** 2
        
        if sol.is_feasible() and self._lex_better(sol.Phi(), backup_sol.Phi()):
            return True
        else:
            sol.mu_sum = backup_sol.mu_sum.copy()
            sol.sigma_sq_sum = backup_sol.sigma_sq_sum.copy()
            sol.assignment = backup_sol.assignment.copy()
            return False
    
    def _lex_better(self, phi1, phi2):
        o1_1, r1, o2_1 = phi1
        o1_2, r2, o2_2 = phi2
        if o1_1 < o1_2 - 1e-9:
            return True
        if o1_1 > o1_2 + 1e-9:
            return False
        if r1 < r2 - 1e-9:
            return True
        if r1 > r2 + 1e-9:
            return False
        return o2_1 < o2_2 - 1e-9
