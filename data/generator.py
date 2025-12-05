"""数据生成器"""
import numpy as np
from typing import List, Tuple
from models.task import Task
from models.server import Server


def generate_tasks(n_tasks: int,
                   mu_range: Tuple[float, float] = (10, 100),
                   cv_range: Tuple[float, float] = (0.3, 0.5),
                   mode: str = "coupled") -> List[Task]:
    """生成任务列表

    Args:
        n_tasks: 任务数量
        mu_range: 期望工作量范围（仅在 mode="coupled" 下使用）
        cv_range: 变异系数范围（仅在 mode="coupled" 下使用）
        mode: 生成模式
            - "coupled": 原模式，σ = μ × CV（默认，保持向后兼容）
            - "bimodal": 双峰模式，μ 与 σ 解耦：
              * ~50% 任务：Type A（大而稳），μ∈[70,90], CV∈[0.08,0.12]
              * ~50% 任务：Type B（小而疯），μ∈[30,50], CV∈[0.70,0.90]

    Returns:
        List[Task]: 任务列表

    设计目的（bimodal模式）：
        - Type A 和 Type B 的期望负载接近（70-90 vs 30-50，整体均值相近）
        - 但方差差异巨大（CV=0.1 vs CV=0.8）
        - 使得"期望均衡"与"风险分散"产生冲突，突出ROSA的风险感知能力
    """
    tasks = []

    if mode == "coupled":
        # 保持原有逻辑：σ = μ × CV（耦合模式）
        for _ in range(n_tasks):
            mu = np.random.uniform(*mu_range)
            cv = np.random.uniform(*cv_range)
            sigma = mu * cv
            tasks.append(Task(mu=mu, sigma=sigma))

    elif mode == "bimodal":
        # 双峰分布：μ 与 σ 解耦
        # Type A（大而稳）：高期望、低方差
        mu_A_range = (70.0, 90.0)
        cv_A_range = (0.08, 0.12)

        # Type B（小而疯）：中等期望、高方差
        mu_B_range = (30.0, 50.0)
        cv_B_range = (0.70, 0.90)

        # 前一半生成 Type A
        n_type_A = n_tasks // 2
        for _ in range(n_type_A):
            mu = np.random.uniform(*mu_A_range)
            cv = np.random.uniform(*cv_A_range)
            sigma = mu * cv
            tasks.append(Task(mu=mu, sigma=sigma))

        # 后一半生成 Type B（奇数任务时，Type B 多一个）
        n_type_B = n_tasks - n_type_A
        for _ in range(n_type_B):
            mu = np.random.uniform(*mu_B_range)
            cv = np.random.uniform(*cv_B_range)
            sigma = mu * cv
            tasks.append(Task(mu=mu, sigma=sigma))

        # 打乱顺序（避免前半段都是 Type A）
        np.random.shuffle(tasks)

    else:
        raise ValueError(f"Unknown mode: {mode}. Expected 'coupled' or 'bimodal'.")

    return tasks


def generate_servers(n_servers: int,
                     f_range: Tuple[float, float] = (100, 200),
                     decision_interval: float = 15.0) -> List[Server]:
    """生成服务器列表

    [核心修改] C = f × T(物理一致性约束)
    [核心修改] f_range扩大至(100,200)以引入强异构性

    强异构性会导致Baseline容易撑爆小服务器,而ROSA能利用大服务器避险

    Args:
        n_servers: 服务器数量
        f_range: CPU频率范围(强异构性,100-200)
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
