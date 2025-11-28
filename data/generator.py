"""数据生成器"""
import numpy as np
from typing import List, Tuple
from models.task import Task
from models.server import Server


def generate_tasks(n_tasks: int,
                   mu_range: Tuple[float, float] = (10, 100),
                   cv_range: Tuple[float, float] = (0.3, 0.5)) -> List[Task]:
    """生成任务列表

    Args:
        n_tasks: 任务数量
        mu_range: 期望工作量范围
        cv_range: 变异系数范围(σ/μ)
    """
    tasks = []
    for _ in range(n_tasks):
        mu = np.random.uniform(*mu_range)
        cv = np.random.uniform(*cv_range)
        sigma = mu * cv
        tasks.append(Task(mu=mu, sigma=sigma))
    return tasks


def generate_servers(n_servers: int,
                     f_range: Tuple[float, float] = (130, 170),
                     decision_interval: float = 15.0) -> List[Server]:
    """生成服务器列表

    [核心修改] C = f × T(物理一致性约束)
    [修正] f_range缩小至(130,170),确保最小容量>平均负载,避免物理瓶颈

    Args:
        n_servers: 服务器数量
        f_range: CPU频率范围(缩小异构性)
        decision_interval: 决策周期(秒)
    """
    servers = []
    for _ in range(n_servers):
        f = np.random.uniform(*f_range)
        C = f * decision_interval  # 关键:容量由处理速率和时间决定
        servers.append(Server(f=f, C=C, L0=0.0))
    return servers


def validate_system_params(servers: List[Server], batch_size: int,
                           mu_avg: float, cv_avg: float, kappa: float,
                           theta: float) -> dict:
    """验证系统参数合理性

    [关键修正] 鲁棒负载应按'分散到M台服务器'计算，而非'聚合到一台'
    分散视角: Sum of Std = √(nM) × σ̄ (方差可加但标准差不可加)
    聚合视角: Aggregated Std = √n × σ̄ (错误假设: 所有任务在一台服务器上)
    """
    n_servers = len(servers)
    total_capacity = sum(s.C for s in servers)

    # 期望负载
    expected_load = batch_size * mu_avg
    rho_expected = expected_load / total_capacity

    # [修正] 分散式鲁棒负载计算
    sigma_avg = mu_avg * cv_avg
    tasks_per_server = batch_size / n_servers
    std_per_server = np.sqrt(tasks_per_server) * sigma_avg
    robust_buffer = n_servers * kappa * std_per_server  # = κ × √(nM) × σ̄
    robust_load = expected_load + robust_buffer
    rho_robust = robust_load / total_capacity

    # θ约束下的有效负载率
    effective_capacity = theta * total_capacity
    rho_effective = robust_load / effective_capacity

    # [诊断] 对比聚合视角(用于验证修正的必要性)
    aggregated_std = np.sqrt(batch_size) * sigma_avg  # √n × σ̄
    aggregated_buffer = kappa * aggregated_std
    buffer_ratio = robust_buffer / aggregated_buffer  # 理论值: √M

    return {
        'total_capacity': total_capacity,
        'expected_load': expected_load,
        'robust_load': robust_load,
        'robust_buffer': robust_buffer,  # 分散视角缓冲
        'aggregated_buffer': aggregated_buffer,  # 聚合视角缓冲(仅供对比)
        'buffer_ratio': buffer_ratio,  # 应约等于√M
        'rho_expected': rho_expected,
        'rho_robust': rho_robust,
        'rho_effective': rho_effective,
        'feasible': rho_effective < 1.0
    }
