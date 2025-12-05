"""NSGA-II (mean-based) risk-neutral baseline.

与 ROSA 接口保持一致，但仅使用期望负载 μ，不考虑方差/鲁棒约束。
"""
import numpy as np
import time
from typing import List, Tuple


class NSGA2MeanSolver:
    """Risk-neutral NSGA-II 调度求解器

    仅使用期望负载(mean)进行优化，不考虑方差/鲁棒性。

    优化目标：
        O1: 期望makespan = max(mean_load_j / f_j)
        O2: 负载不均衡 = std(mean_load_j)
        O3: 已禁用（原为常数，不参与优化）

    Args:
        n_pop: 种群大小
        g_max: 最大迭代次数
        t_max: 最大运行时间(秒)
        p_c: 交叉概率
        p_m: 变异概率
        n_elite: 精英保留数量
        w1: O1权重(makespan)
        w2: O2权重(负载均衡)
        w3: 保留参数(不使用，仅为接口兼容)
    """

    def __init__(self,
                 n_pop: int = 50,
                 g_max: int = 100,
                 t_max: float = 0.1,
                 p_c: float = 0.8,
                 p_m: float = 0.1,
                 n_elite: int = 2,
                 w1: float = 0.5,
                 w2: float = 0.3,
                 w3: float = 0.2):  # w3不使用，仅为接口兼容
        self.n_pop = n_pop
        self.g_max = g_max
        self.t_max = t_max
        self.p_c = p_c
        self.p_m = p_m
        self.n_elite = n_elite
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

        # 归一化边界
        self.O1_min = self.O1_max = 0.0
        self.O2_min = self.O2_max = 0.0
        self.O3_min = self.O3_max = 0.0

    def solve_batch(self, tasks: List, servers: List) -> List[int]:
        """处理一个批次的任务，返回任务->服务器分配"""
        n_tasks = len(tasks)
        n_servers = len(servers)

        task_list = [{'mu': t.mu} for t in tasks]
        server_list = [{'f': s.f, 'C': s.C, 'L0': s.L0} for s in servers]

        # 初始化种群
        population = self._initialize_population(task_list, server_list)

        start_time = time.time()
        g = 0
        fitness = None
        while time.time() - start_time < self.t_max and g < self.g_max:
            # 每代刷新归一化边界并评估父代
            self._init_normalization_bounds(population, task_list, server_list)
            fitness = [self._evaluate_mean(ind, task_list, server_list)
                       for ind in population]

            parents = self._tournament_selection(population, fitness)
            offspring = self._uniform_crossover(parents)
            offspring = [self._random_mutation(ind, n_servers)
                         for ind in offspring]

            g += 1
            offspring_fitness = [self._evaluate_mean(ind, task_list, server_list)
                                 for ind in offspring]

            population, fitness = self._elitist_replacement(
                population, fitness, offspring, offspring_fitness
            )

        # 终止时再刷新一次归一化并评估
        self._init_normalization_bounds(population, task_list, server_list)
        fitness = [self._evaluate_mean(ind, task_list, server_list)
                   for ind in population]

        # 选出最优（最小适应度）
        best_idx = int(np.argmin(fitness))
        return population[best_idx]

    # ---------- 初始化 ----------
    def _initialize_population(self, tasks: List[dict], servers: List[dict]) -> List[List[int]]:
        """简单的基于 μ 的贪心初始化 + 随机多样化"""
        n_tasks = len(tasks)
        population = []

        # 按 μ 降序
        sorted_tasks = sorted([(i, tasks[i]['mu']) for i in range(n_tasks)],
                              key=lambda x: -x[1])

        elite = self._greedy_assign(sorted_tasks, tasks, servers, randomize=False)
        population.append(elite)

        for _ in range(self.n_pop - 1):
            ind = self._greedy_assign(sorted_tasks, tasks, servers, randomize=True)
            population.append(ind)

        return population

    def _greedy_assign(self, sorted_tasks: List[Tuple[int, float]], tasks: List[dict],
                       servers: List[dict], randomize: bool = False,
                       top_k: int = 3) -> List[int]:
        n_servers = len(servers)
        assignment = [0] * len(tasks)
        mean_load = np.array([s['L0'] for s in servers])

        for task_idx, _ in sorted_tasks:
            mu_i = tasks[task_idx]['mu']

            # 选择期望负载最小的服务器（可带随机扰动）
            if randomize:
                # 选择 top_k 最小中的随机一个
                order = np.argsort(mean_load)
                k = min(top_k, n_servers)
                j_star = int(np.random.choice(order[:k]))
            else:
                j_star = int(np.argmin(mean_load))

            assignment[task_idx] = j_star
            mean_load[j_star] += mu_i

        return assignment

    # ---------- 目标评估 ----------
    def _compute_mean_loads(self, assignment: List[int], tasks: List[dict],
                            servers: List[dict]) -> np.ndarray:
        n_servers = len(servers)
        mean_sum = np.array([s['L0'] for s in servers])
        for i, j in enumerate(assignment):
            mean_sum[j] += tasks[i]['mu']
        return mean_sum

    def _compute_mean_objectives(self, assignment: List[int], tasks: List[dict],
                                 servers: List[dict]) -> Tuple[float, float, float]:
        mean_load = self._compute_mean_loads(assignment, tasks, servers)
        f_j = np.array([s['f'] for s in servers])

        # O1: 期望 makespan（用 mean_load/f）
        O1 = float(np.max(mean_load / f_j))

        # O2: 期望负载标准差（衡量负载均衡性）
        O2 = float(np.std(mean_load))

        # O3: 占位符（不参与优化，仅为保持接口一致）
        # 注：sum(mean_load)对固定任务集是常数，已从优化目标中移除
        O3 = 0.0

        return O1, O2, O3

    def _init_normalization_bounds(self, population: List[List[int]],
                                   tasks: List[dict], servers: List[dict]):
        O1_vals, O2_vals, O3_vals = [], [], []
        for ind in population:
            O1, O2, O3 = self._compute_mean_objectives(ind, tasks, servers)
            O1_vals.append(O1)
            O2_vals.append(O2)
            O3_vals.append(O3)
        self.O1_min, self.O1_max = min(O1_vals), max(O1_vals)
        self.O2_min, self.O2_max = min(O2_vals), max(O2_vals)
        self.O3_min, self.O3_max = min(O3_vals), max(O3_vals)

    def _evaluate_mean(self, assignment: List[int], tasks: List[dict],
                       servers: List[dict]) -> float:
        O1, O2, _ = self._compute_mean_objectives(assignment, tasks, servers)

        O1_norm = (O1 - self.O1_min) / max(self.O1_max - self.O1_min, 1e-6)
        O2_norm = (O2 - self.O2_min) / max(self.O2_max - self.O2_min, 1e-6)

        # 只使用O1和O2，重新归一化权重（忽略w3）
        w_sum = self.w1 + self.w2
        w1_norm = self.w1 / max(w_sum, 1e-6)
        w2_norm = self.w2 / max(w_sum, 1e-6)

        return w1_norm * O1_norm + w2_norm * O2_norm

    # ---------- GA 操作 ----------
    def _tournament_selection(self, population: List[List[int]],
                              fitness: List[float]) -> List[List[int]]:
        parents = []
        n = len(population)
        for _ in range(n):
            i, j = np.random.randint(n, size=2)
            winner = population[i] if fitness[i] < fitness[j] else population[j]
            parents.append(winner.copy())
        return parents

    def _uniform_crossover(self, parents: List[List[int]]) -> List[List[int]]:
        offspring = []
        n = len(parents)
        for k in range(0, n - 1, 2):
            p1, p2 = parents[k], parents[k + 1]
            c1, c2 = p1.copy(), p2.copy()
            if np.random.random() < self.p_c:
                for i in range(len(p1)):
                    if np.random.random() < 0.5:
                        c1[i], c2[i] = c2[i], c1[i]
            offspring.extend([c1, c2])
        if n % 2 == 1:
            offspring.append(parents[-1].copy())
        return offspring

    def _random_mutation(self, individual: List[int], n_servers: int) -> List[int]:
        """仅随机变异：把一个任务换到另一台随机服务器"""
        if np.random.random() > self.p_m:
            return individual

        ind = individual.copy()
        i = np.random.randint(len(ind))
        choices = [j for j in range(n_servers) if j != ind[i]]
        ind[i] = int(np.random.choice(choices)) if choices else ind[i]
        return ind

    def _elitist_replacement(self, population: List[List[int]], fitness: List[float],
                             offspring: List[List[int]], offspring_fitness: List[float]
                             ) -> Tuple[List[List[int]], List[float]]:
        combined = list(zip(population + offspring, fitness + offspring_fitness))
        combined.sort(key=lambda x: x[1])
        new_pop = [x[0] for x in combined[:self.n_pop]]
        new_fit = [x[1] for x in combined[:self.n_pop]]
        return new_pop, new_fit
