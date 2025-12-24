"""蒙特卡洛测试: 计算 CVR(Capacity Violation Rate)"""
import numpy as np
from typing import List


def monte_carlo_cvr(assignment: List[int], tasks: List, servers: List,
                    n_samples: int = 10000) -> float:
    """蒙特卡洛模拟计算容量违规率

    对于每次模拟:
    1. 从 N(μ_i, σ_i²) 采样每个任务的实际工作量
    2. 计算每个服务器的实际负载
    3. 检查是否有服务器超过容量

    Returns:
        cvr: 容量违规率(有任意服务器过载的模拟比例)
    """
    n_servers = len(servers)

    mu = np.array([t.mu for t in tasks])
    sigma = np.array([t.sigma for t in tasks])
    C = np.array([s.C for s in servers])
    L0 = np.array([s.L0 for s in servers])

    violations = 0

    for _ in range(n_samples):
        actual_workload = np.random.normal(mu, sigma)
        actual_workload = np.maximum(actual_workload, 0)

        actual_load = L0.copy()
        for i, j in enumerate(assignment):
            actual_load[j] += actual_workload[i]

        if np.any(actual_load > C):
            violations += 1

    return violations / n_samples
