"""目标函数计算"""
import numpy as np
from typing import List


def compute_objectives(assignment: List[int], tasks: List, servers: List,
                       kappa: float, theta: float) -> dict:
    """计算三个目标函数值

    Returns:
        dict: {'O1': ..., 'O2': ..., 'O3': ..., 'robust_loads': [...]}
    """
    n_servers = len(servers)

    mu_sum = np.array([s.L0 for s in servers])
    var_sum = np.zeros(n_servers)

    for i, j in enumerate(assignment):
        mu_sum[j] += tasks[i].mu
        var_sum[j] += tasks[i].sigma ** 2

    robust_load = mu_sum + kappa * np.sqrt(var_sum)

    f_j = np.array([s.f for s in servers])

    O1 = float(np.max(robust_load / f_j))

    # 与论文一致：O2 为鲁棒负载的标准差（不除以容量）
    O2 = float(np.std(robust_load))

    C_j = np.array([s.C for s in servers])
    gap_j = theta * C_j - robust_load
    epsilon = 0.01 * np.mean(C_j)
    O3 = sum(
        tasks[i].get_delta(kappa) / max(gap_j[j], epsilon)
        for i, j in enumerate(assignment)
    )

    return {
        'O1': O1,
        'O2': O2,
        'O3': O3,
        'robust_loads': robust_load.tolist()
    }
