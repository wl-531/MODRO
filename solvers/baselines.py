"""Baseline调度算法集合

包含:
1. Deterministic Greedy (DG): 仅基于期望负载的贪心
2. Variance-Aware Greedy (VAG): 考虑方差的贪心
"""
import numpy as np
from typing import List


def deterministic_greedy(tasks, servers):
    """Deterministic-Greedy基准算法 [MOVED]

    按期望负载E[L]=L0+Σμ最小化原则进行贪婪分配,不考虑方差σ
    [修复] 使用利用率（load/capacity）而非绝对负载，以支持异构服务器

    Args:
        tasks: 任务列表，每个任务有 .mu (期望工作量) 属性
        servers: 服务器列表，每个服务器有 .L0 (已有负载), .C (容量) 属性

    Returns:
        assignment: List[int], assignment[i] = j 表示任务i分配给服务器j
    """
    n_tasks = len(tasks)
    n_servers = len(servers)
    assignment = []

    current_load = np.array([s.L0 for s in servers])
    capacities = np.array([s.C for s in servers])  # 获取服务器容量

    for i in range(n_tasks):
        mu_i = tasks[i].mu

        # [修复] 选择当前利用率最小的服务器（支持异构）
        utilization = current_load / capacities
        j_min = int(np.argmin(utilization))

        assignment.append(j_min)
        current_load[j_min] += mu_i

    return assignment


def variance_aware_greedy(tasks, servers, lambda_=1.0):  # NEW
    """Variance-Aware Greedy基准算法 (VAG)

    在选择服务器时同时考虑期望负载和方差聚合，使用打分函数:
    score_j = (mean_new_j / C_j) + lambda_ * (std_new_j / C_j)
    [修复] 归一化到容量以支持异构服务器

    Args:
        tasks: 任务列表，每个任务有 .mu (期望) 和 .sigma (标准差) 属性
        servers: 服务器列表，每个服务器有 .L0 (已有负载), .C (容量) 属性
        lambda_: 风险权重系数，控制方差在决策中的权重 (默认1.0)

    Returns:
        assignment: List[int], assignment[i] = j 表示任务i分配给服务器j
    """
    n_tasks = len(tasks)
    n_servers = len(servers)
    assignment = []

    # 维护期望负载和方差聚合
    current_mean = np.array([s.L0 for s in servers])
    current_var = np.zeros(n_servers)  # 初始方差为0
    capacities = np.array([s.C for s in servers])  # 获取服务器容量

    for i in range(n_tasks):
        mu_i = tasks[i].mu
        sigma_i = tasks[i].sigma
        var_i = sigma_i ** 2

        # 枚举所有服务器，计算新的打分函数
        scores = np.zeros(n_servers)
        for j in range(n_servers):
            mean_new_j = current_mean[j] + mu_i
            var_new_j = current_var[j] + var_i
            std_new_j = np.sqrt(var_new_j)

            # [修复] 核心打分函数: 归一化到容量（支持异构）
            normalized_mean = mean_new_j / capacities[j]
            normalized_std = std_new_j / capacities[j]
            scores[j] = normalized_mean + lambda_ * normalized_std

        # 选择score最小的服务器
        j_min = int(np.argmin(scores))
        assignment.append(j_min)

        # 更新该服务器的状态
        current_mean[j_min] += mu_i
        current_var[j_min] += var_i

    return assignment
