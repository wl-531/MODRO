"""服务器状态监控模块 - 期望 vs 实际负载对比"""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List


@dataclass
class ServerSnapshot:
    """单台服务器的快照 - 期望 vs 实际"""
    batch_idx: int
    algo: str
    server_id: int
    capacity: float       # 容量
    mu_j: float           # 期望负载
    sigma_j: float        # 聚合标准差
    actual_load: float    # 实际负载（采样）
    delta: float          # 波动量 = actual - mu
    overload: bool        # 是否实际超载


class ServerMonitor:
    """监控服务器状态 - 期望 vs 实际负载对比"""

    def __init__(self):
        self.snapshots = []  # 所有批次的服务器快照

    def record_batch(self, batch_idx: int, algo_name: str,
                     assignment: List[int], tasks: List, servers: List):
        """记录单批次所有服务器状态（必须采样实际负载）"""
        n_servers = len(servers)

        # 采样实际工作量
        mu = np.array([t.mu for t in tasks])
        sigma = np.array([t.sigma for t in tasks])
        actual_workload = np.random.normal(mu, sigma)
        actual_workload = np.maximum(actual_workload, 0)

        # 每台服务器的指标
        for j in range(n_servers):
            # 找到分配到服务器j的任务
            task_indices = [i for i, server_id in enumerate(assignment) if server_id == j]

            # 期望负载
            mu_j = servers[j].L0 + sum(tasks[i].mu for i in task_indices)

            # 聚合标准差
            sigma_j = np.sqrt(sum(tasks[i].sigma**2 for i in task_indices))

            # 实际负载
            actual_load_j = servers[j].L0 + sum(actual_workload[i] for i in task_indices)

            # 波动量
            delta = actual_load_j - mu_j

            # 是否超载
            overload = actual_load_j > servers[j].C

            # 保存快照
            snapshot = ServerSnapshot(
                batch_idx=batch_idx,
                algo=algo_name,
                server_id=j,
                capacity=servers[j].C,
                mu_j=mu_j,
                sigma_j=sigma_j,
                actual_load=actual_load_j,
                delta=delta,
                overload=overload
            )
            self.snapshots.append(snapshot)

    def get_summary(self) -> dict:
        """汇总统计：期望 vs 实际波动分析"""
        algos = sorted(set(s.algo for s in self.snapshots))
        summary = {}

        for algo in algos:
            algo_snaps = [s for s in self.snapshots if s.algo == algo]

            # 波动统计
            deltas = [s.delta for s in algo_snaps]
            abs_deltas = [abs(d) for d in deltas]

            summary[algo] = {
                'avg_abs_delta': np.mean(abs_deltas),
                'max_abs_delta': np.max(abs_deltas),
                'std_delta': np.std(deltas),
                'overload_count': sum(1 for s in algo_snaps if s.overload)
            }

        return summary

    def export_csv(self, filepath: str):
        """导出详细数据到CSV - 期望 vs 实际"""
        rows = []
        for s in self.snapshots:
            rows.append({
                'batch_idx': s.batch_idx,
                'algo': s.algo,
                'server_id': s.server_id,
                'capacity': s.capacity,
                'mu_j': s.mu_j,
                'sigma_j': s.sigma_j,
                'actual_load': s.actual_load,
                'delta': s.delta,
                'overload': int(s.overload)
            })

        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)
        print(f"Saved diagnostics to {filepath}")

    def print_summary(self):
        """打印期望 vs 实际波动分析"""
        summary = self.get_summary()
        if not summary:
            print("No data to summarize")
            return

        print("\n[期望 vs 实际 波动分析]")
        algos = sorted(summary.keys())

        # 表头
        header = "Algo  | Avg|Δ| | Max|Δ| | Std(Δ) | Overload Count"
        print(header)
        print("-" * len(header))

        # 每个算法一行
        for algo in algos:
            s = summary[algo]
            print(f"{algo.upper():6s}| {s['avg_abs_delta']:7.1f} | "
                  f"{s['max_abs_delta']:6.1f} | {s['std_delta']:6.1f} | "
                  f"{s['overload_count']:14d}")

        print("\n解读: Avg|Δ| 越小，实际负载波动越小（更稳定）")
