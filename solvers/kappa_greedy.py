"""κ-Greedy: 基于鲁棒负载的贪心调度算法

核心思想：
- 用鲁棒负载 L̂_j = μ_j + κ·σ_j 代替期望负载决策
- κ 由 Cantelli 不等式确定：κ = √(1/α - 1)
- 理论保证：CVR ≤ α

与 VAG 的关键区别：
- VAG 用线性近似 μ + λ·Σσ_i
- κ-Greedy 用精确聚合 μ + κ·√(Σσ²_i)
"""
import numpy as np
from typing import List


def kappa_greedy(tasks: List, servers: List, kappa: float) -> List[int]:
    """κ-Greedy 调度算法
    
    Args:
        tasks: 任务列表，每个任务有 .mu 和 .sigma 属性
        servers: 服务器列表，每个服务器有 .L0, .C 属性
        kappa: 不确定性系数，由 Cantelli 不等式确定
               κ = √(1/α - 1)，α=0.15 时 κ≈2.38
    
    Returns:
        assignment: List[int]，assignment[i] = j 表示任务 i 分配给服务器 j
    """
    n_tasks = len(tasks)
    n_servers = len(servers)
    
    # 初始化每个服务器的状态
    mu_sum = np.array([s.L0 for s in servers])      # 期望负载
    var_sum = np.zeros(n_servers)                    # 方差累积
    capacities = np.array([s.C for s in servers])   # 容量
    
    assignment = []
    
    for i in range(n_tasks):
        mu_i = tasks[i].mu
        var_i = tasks[i].sigma ** 2
        
        # 计算分配到每个服务器后的鲁棒利用率
        new_mu = mu_sum + mu_i
        new_var = var_sum + var_i
        new_std = np.sqrt(new_var)
        new_robust = new_mu + kappa * new_std
        new_util = new_robust / capacities
        
        # 选择鲁棒利用率最小的服务器
        j_star = int(np.argmin(new_util))
        
        assignment.append(j_star)
        
        # 更新选中服务器的状态
        mu_sum[j_star] += mu_i
        var_sum[j_star] += var_i
    
    return assignment
