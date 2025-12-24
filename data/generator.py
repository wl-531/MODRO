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
            - "coupled": 原模式，sigma = mu * CV（默认，保持向后兼容）
            - "bimodal": 双峰模式，mu-σ 解耦
              * ~50% 任务：Type A（稳定），mu∈[55,65], CV∈[0.08,0.12]
              * ~50% 任务：Type B（疯狂），mu∈[55,65], CV∈[0.70,0.90]
            - "multiclass": 多类型任务模式
              * 25% 视频/重计算：mu∈[80,120], CV∈[0.30,0.50]
              * 25% IO/抖动型：mu∈[50,70], CV∈[0.60,0.90]（高风险）
              * 30% 语音/轻量：mu∈[20,40], CV∈[0.08,0.15]
              * 20% 后台/管理：mu∈[5,15], CV∈[0.02,0.08]

    Returns:
        List[Task]
    """
    tasks = []

    if mode == "coupled":
        # 原逻辑：sigma = mu * CV（耦合模式）
        for _ in range(n_tasks):
            mu = np.random.uniform(*mu_range)
            cv = np.random.uniform(*cv_range)
            sigma = mu * cv
            tasks.append(Task(mu=mu, sigma=sigma))

    elif mode == "bimodal":
        # 双峰：mu-σ 解耦，mu 相近但方差差异极大
        mu_A_range = (55.0, 65.0)
        cv_A_range = (0.58, 0.92)

        mu_B_range = (55.0, 65.0)
        cv_B_range = (1.80, 2.60)

        # 按 1:2 比例生成：A 占约 1/3，B 占约 2/3（余数给 B）
        n_type_A = n_tasks // 3
        for _ in range(n_type_A):
            mu = np.random.uniform(*mu_A_range)
            cv = np.random.uniform(*cv_A_range)
            sigma = mu * cv
            tasks.append(Task(mu=mu, sigma=sigma))

        # 其余生成 Type B
        n_type_B = n_tasks - n_type_A
        for _ in range(n_type_B):
            mu = np.random.uniform(*mu_B_range)
            cv = np.random.uniform(*cv_B_range)
            sigma = mu * cv
            tasks.append(Task(mu=mu, sigma=sigma))

        # 打乱顺序
        np.random.shuffle(tasks)

    elif mode == "multiclass":
        # 多类型任务：4种不同特征的任务类型，模拟真实边缘计算场景
        task_types = [
            ("video",   0.25, (80, 120), (0.30, 0.50)),  # 视频/重计算：μ高、σ中
            ("io",      0.25, (50, 80),  (0.60, 0.90)),  # IO/抖动型：μ中、σ高
            ("audio",   0.20, (10, 60),  (0.08, 0.15)),  # 语音/轻量：μ低、σ低
            ("backend", 0.10, (5, 15),   (0.02, 0.08)),  # 后台/管理：μ极低、σ极低
        ]

        for type_name, ratio, mu_range_type, cv_range_type in task_types:
            n_type = int(n_tasks * ratio)
            for _ in range(n_type):
                mu = np.random.uniform(*mu_range_type)
                cv = np.random.uniform(*cv_range_type)
                sigma = mu * cv
                tasks.append(Task(mu=mu, sigma=sigma))

        # 补齐因取整丢失的任务（随机选一个类型）
        while len(tasks) < n_tasks:
            type_name, ratio, mu_range_type, cv_range_type = task_types[np.random.randint(len(task_types))]
            mu = np.random.uniform(*mu_range_type)
            cv = np.random.uniform(*cv_range_type)
            sigma = mu * cv
            tasks.append(Task(mu=mu, sigma=sigma))

        # 打乱顺序
        np.random.shuffle(tasks)

    else:
        raise ValueError(f"Unknown mode: {mode}. Expected 'coupled' or 'bimodal'.")

    return tasks


def generate_servers(n_servers: int,
                     f_range: Tuple[float, float] = (100, 200),
                     decision_interval: float = 15.0) -> List[Server]:
    """生成服务器列表"""
    servers = []
    for _ in range(n_servers):
        f = np.random.uniform(*f_range)
        C = f * decision_interval  # 容量 = 处理速率 * 决策周期
        servers.append(Server(f=f, C=C, L0=0.0))
    return servers


def validate_system_params(servers: List[Server], batch_size: int,
                           mu_avg: float, cv_avg: float, kappa: float,
                           theta: float) -> dict:
    """验证系统参数合理性"""
    n_servers = len(servers)
    total_capacity = sum(s.C for s in servers)

    # 期望负载
    expected_load = batch_size * mu_avg
    rho_expected = expected_load / total_capacity

    # 鲁棒负载（分散视角）
    sigma_avg = mu_avg * cv_avg
    tasks_per_server = batch_size / n_servers
    std_per_server = np.sqrt(tasks_per_server) * sigma_avg
    robust_buffer = n_servers * kappa * std_per_server
    robust_load = expected_load + robust_buffer
    rho_robust = robust_load / total_capacity

    # theta 约束下的有效负载率
    effective_capacity = theta * total_capacity
    rho_effective = robust_load / effective_capacity

    # 对比聚合视角（诊断用）
    aggregated_std = np.sqrt(batch_size) * sigma_avg
    aggregated_buffer = kappa * aggregated_std
    buffer_ratio = robust_buffer / aggregated_buffer  # 理论值≈sqrt(M)

    return {
        'total_capacity': total_capacity,
        'expected_load': expected_load,
        'robust_load': robust_load,
        'robust_buffer': robust_buffer,
        'aggregated_buffer': aggregated_buffer,
        'buffer_ratio': buffer_ratio,
        'rho_expected': rho_expected,
        'rho_robust': rho_robust,
        'rho_effective': rho_effective,
        'feasible': rho_effective < 1.0
    }
