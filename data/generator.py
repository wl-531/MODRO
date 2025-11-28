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

    注意: 鲁棒负载利用方差可加性计算,而非简单累加边际鲁棒负载
    """
    total_capacity = sum(s.C for s in servers)

    # 期望负载
    expected_load = batch_size * mu_avg
    rho_expected = expected_load / total_capacity

    # [修正] 鲁棒负载: 利用方差可加性
    # Var(L̃) = Σ σ_i² = n × σ̄²
    # Std(L̃) = √(n × σ̄²) = √n × σ̄
    sigma_avg = mu_avg * cv_avg
    total_std = np.sqrt(batch_size) * sigma_avg  # √n × σ̄
    robust_load = expected_load + kappa * total_std
    rho_robust = robust_load / total_capacity

    # θ约束下的有效负载率
    effective_capacity = theta * total_capacity
    rho_effective = robust_load / effective_capacity

    return {
        'total_capacity': total_capacity,
        'expected_load': expected_load,
        'robust_load': robust_load,
        'total_std': total_std,  # 便于调试
        'rho_expected': rho_expected,
        'rho_robust': rho_robust,
        'rho_effective': rho_effective,
        'feasible': rho_effective < 1.0
    }
