"""统一指标计算（按文档定义）"""
import numpy as np
from typing import List, Dict
from models.task import Task
from models.server import Server


def compute_metrics_unified(
    assignment: List[int],
    tasks: List[Task],
    servers: List[Server],
    kappa: float,
    eps_div: float = 1e-6,
    tol_feas: float = 1e-9
) -> Dict[str, float]:
    """
    统一计算所有物理指标（文档定义）
    
    Args:
        assignment: List[int], assignment[i] = j 表示 tasks[i] 分配到 servers[j]
        tasks: 任务列表（按顺序，assignment 下标对应 tasks 列表位置）
        servers: 服务器列表
        kappa: 风险系数
        eps_div: 除零保护常数
        tol_feas: 可行性容差
    
    Returns:
        dict with keys: 'feasible', 'U_max', 'O1', 'R_sum', 'O2',
                        'L_hat' (array), 'Gap' (array), 'RD' (array)
    """
    n_tasks = len(tasks)
    m = len(servers)
    
    # 断言检查
    assert len(assignment) == n_tasks, "assignment 长度必须等于 tasks 数量"
    assert all(0 <= a < m for a in assignment), "assignment 值必须在 [0, m) 范围内"
    
    # 初始化
    C = np.array([s.C for s in servers])
    L0 = np.array([s.L0 for s in servers])
    mu_sum = np.zeros(m)
    sigma_sq_sum = np.zeros(m)
    
    # 累加
    for i, j in enumerate(assignment):
        mu_sum[j] += tasks[i].mu
        sigma_sq_sum[j] += tasks[i].sigma ** 2
    
    # 计算状态量
    sigma_j = np.sqrt(np.maximum(sigma_sq_sum, 0))
    L_hat = L0 + mu_sum + kappa * sigma_j
    Gap = C - L_hat
    RD = sigma_j / np.maximum(Gap, eps_div)
    
    # 计算指标
    feasible = bool(np.all(Gap >= -tol_feas))
    U_max = float(np.max(L_hat / C))
    O1 = float(np.max(L_hat))
    R_sum = float(np.sum(RD))
    L_bar = np.mean(L_hat)
    O2 = float(np.sum((L_hat - L_bar) ** 2))
    
    return {
        'feasible': feasible,
        'U_max': U_max,
        'O1': O1,
        'R_sum': R_sum,
        'O2': O2,
        'L_hat': L_hat,
        'Gap': Gap,
        'RD': RD
    }
