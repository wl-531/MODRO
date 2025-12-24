"""ROSA: Robust Online Scheduling Algorithm

核心创新:
1. Robust-FFD 初始化: 按 Δ_i 降序排列,Best-Fit 分配
2. De-risking Mutation: 识别风险热点并迁移
3. 动态惩罚机制: 惩罚系数随迭代增长
"""
import numpy as np
import time
from typing import List, Tuple


class ROSASolver:
    """ROSA 求解器"""

    def __init__(self,
                 kappa: float,
                 theta: float,
                 n_pop: int = 50,
                 g_max: int = 100,
                 t_max: float = 0.1,
                 p_c: float = 0.8,
                 p_m: float = 0.1,
                 p_risk: float = 0.7,
                 n_elite: int = 2,
                 lambda_0: float = 1.0,
                 beta: float = 0.1,
                 w1: float = 0.5,
                 w2: float = 0.3,
                 w3: float = 0.2):
        self.kappa = kappa
        self.theta = theta
        self.n_pop = n_pop
        self.g_max = g_max
        self.t_max = t_max
        self.p_c = p_c
        self.p_m = p_m
        self.p_risk = p_risk
        self.n_elite = n_elite
        self.lambda_0 = lambda_0
        self.beta = beta
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

        # 归一化边界
        self.O1_min = self.O1_max = 0.0
        self.O2_min = self.O2_max = 0.0
        self.O3_min = self.O3_max = 0.0

    def solve_batch(self, tasks: List, servers: List) -> List[int]:
        """处理一个批次的任务

        Args:
            tasks: 任务列表,每个任务有属性 mu, sigma
            servers: 服务器列表,每个服务器有属性 f, C, L0

        Returns:
            assignment: 分配方案,assignment[i] = j 表示任务i分配给服务器j
        """
        n_tasks = len(tasks)
        n_servers = len(servers)

        task_dicts = [{'mu': t.mu, 'sigma': t.sigma} for t in tasks]
        server_dicts = [{'f': s.f, 'C': s.C, 'L0': s.L0} for s in servers]

        # 1. Robust-FFD 初始化
        population = self._initialize_population(task_dicts, server_dicts)

        # ===== 诊断输出 START =====
        n_servers = len(servers)
        C_arr = np.array([s['C'] for s in server_dicts])

        # 检查精英解的可行性
        elite = population[0]
        _, _, elite_load = self._compute_server_loads(elite, task_dicts, server_dicts)
        elite_util = elite_load / C_arr
        elite_feasible = all(elite_load <= self.theta * C_arr)

        # 统计初始种群可行解数量
        n_feasible_init = sum(
            1 for ind in population
            if all(self._compute_server_loads(ind, task_dicts, server_dicts)[2] <= self.theta * C_arr)
        )

        print(f"[DIAG-INIT] 精英解可行: {elite_feasible}, max_util={max(elite_util):.3f}, "
              f"初始可行解数: {n_feasible_init}/{len(population)}")
        # ===== 诊断输出 END =====

        # 0. 保存初始精英解(防止进化破坏优质初始解)
        elite = population[0]
        _, _, elite_load = self._compute_server_loads(elite, task_dicts, server_dicts)
        C_j_arr = np.array([s['C'] for s in server_dicts])
        # [修复缺陷1] 即使不可行也保存，后续会比较违规程度
        fallback_solution = elite
        fallback_violation = np.sum(np.maximum(0, elite_load - self.theta * C_j_arr) ** 2)

        # 迭代进化
        start_time = time.time()
        g = 0
        fitness = None
        while time.time() - start_time < self.t_max and g < self.g_max:
            # 每代刷新归一化边界并重评父代（与论文描述一致）
            self._init_normalization_bounds(population, task_dicts, server_dicts)
            fitness = [self._evaluate(ind, task_dicts, server_dicts, g=g)
                       for ind in population]

            parents = self._tournament_selection(population, fitness)
            offspring = self._uniform_crossover(parents)
            offspring = [self._hybrid_mutation(ind, task_dicts, server_dicts)
                         for ind in offspring]

            g += 1
            offspring_fitness = [self._evaluate(ind, task_dicts, server_dicts, g=g)
                                 for ind in offspring]

            population, fitness = self._elitist_replacement(
                population, fitness, offspring, offspring_fitness
            )

        # 进化结束后再刷新一次归一化边界并评估，用于最终选择
        self._init_normalization_bounds(population, task_dicts, server_dicts)
        fitness = [self._evaluate(ind, task_dicts, server_dicts, g=g)
                   for ind in population]

        # 获取最优解
        best = self._get_best_feasible(population, fitness, task_dicts, server_dicts)

        # [修复缺陷2] 处理极端情况：如果best是None（理论上不会），返回fallback
        if best is None:
            return fallback_solution

        _, _, best_load = self._compute_server_loads(best, task_dicts, server_dicts)
        best_violation = np.sum(np.maximum(0, best_load - self.theta * C_j_arr) ** 2)

        # [修复缺陷3] 比较违规程度，选择更优的解
        # 1. 如果best可行，直接返回
        # ===== 诊断输出 START =====
        # 统计最终种群可行解数量
        n_feasible_final = sum(
            1 for ind in population
            if all(self._compute_server_loads(ind, task_dicts, server_dicts)[2] <= self.theta * C_arr)
        )

        # 检查最终解的状态
        _, _, best_load = self._compute_server_loads(best, task_dicts, server_dicts)
        best_util = best_load / C_arr
        best_feasible = all(best_load <= self.theta * C_arr)

        # 计算 gap 分布（用于判断 O3 是否有效）
        gap_arr = self.theta * C_arr - best_load
        n_negative_gap = sum(1 for g in gap_arr if g < 0)

        print(f"[DIAG-FINAL] 最终解可行: {best_feasible}, max_util={max(best_util):.3f}, "
              f"最终可行解数: {n_feasible_final}/{len(population)}, "
              f"负gap服务器数: {n_negative_gap}/{n_servers}")
        # ===== 诊断输出 END =====

        if best_violation < 1e-6:
            return best
        # 2. 如果都不可行，返回违规更小的
        if best_violation > fallback_violation:
            return fallback_solution
        else:
            return best

    def _initialize_population(self, tasks: List[dict], servers: List[dict]) -> List[List[int]]:
        """Robust-FFD 初始化

        排序依据: Δ_i = μ_i + κσ_i(保守估计,用于排序)
        可行性检查: 精确计算聚合标准差 √(Σσ_i²)(利用方差可加性)
        """
        n_tasks = len(tasks)
        population = []

        # 按 Δ_i 降序排列
        deltas = [(i, tasks[i]['mu'] + self.kappa * tasks[i]['sigma'])
                  for i in range(n_tasks)]
        sorted_tasks = sorted(deltas, key=lambda x: -x[1])

        # 精英个体
        elite = self._greedy_assign(sorted_tasks, tasks, servers, randomize=False)
        population.append(elite)

        # 多样性个体
        for _ in range(self.n_pop - 1):
            ind = self._greedy_assign(sorted_tasks, tasks, servers, randomize=True)
            population.append(ind)

        return population

    def _greedy_assign(self, sorted_tasks: List[Tuple], tasks: List[dict],
                       servers: List[dict], randomize: bool = False,
                       top_k: int = 3) -> List[int]:
        """贪婪分配

        分别跟踪 mu_sum 和 var_sum,利用方差可加性精确计算鲁棒负载
        """
        n_servers = len(servers)
        assignment = [0] * len(tasks)

        current_mu_sum = np.array([s['L0'] for s in servers])
        current_var_sum = np.zeros(n_servers)

        for task_idx, delta_i in sorted_tasks:
            mu_i = tasks[task_idx]['mu']
            sigma_i = tasks[task_idx]['sigma']

            feasible = []
            for j in range(n_servers):
                new_mu_sum = current_mu_sum[j] + mu_i
                new_var_sum = current_var_sum[j] + sigma_i ** 2
                new_robust = new_mu_sum + self.kappa * np.sqrt(new_var_sum)

                if new_robust <= self.theta * servers[j]['C']:
                    utilization = new_robust / servers[j]['C']
                    feasible.append((j, utilization))

            if feasible:
                feasible.sort(key=lambda x: x[1])
                if randomize and len(feasible) > 1:
                    k = min(top_k, len(feasible))
                    chosen = feasible[np.random.randint(k)]
                else:
                    chosen = feasible[0]
                j_star = chosen[0]
            else:
                # 论文算法2回退：选择鲁棒利用率最小的服务器
                C_arr = np.array([s['C'] for s in servers])
                new_mu = current_mu_sum + mu_i
                new_std = np.sqrt(current_var_sum + sigma_i ** 2)
                new_robust = new_mu + self.kappa * new_std
                utilization = new_robust / C_arr
                j_star = int(np.argmin(utilization))

            assignment[task_idx] = j_star
            current_mu_sum[j_star] += mu_i
            current_var_sum[j_star] += sigma_i ** 2

        return assignment

    def _compute_server_loads(self, assignment: List[int], tasks: List[dict],
                              servers: List[dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """计算服务器负载统计量"""
        n_servers = len(servers)
        mu_sum = np.array([s['L0'] for s in servers])
        var_sum = np.zeros(n_servers)

        for i, j in enumerate(assignment):
            mu_sum[j] += tasks[i]['mu']
            var_sum[j] += tasks[i]['sigma'] ** 2

        robust_load = mu_sum + self.kappa * np.sqrt(var_sum)
        return mu_sum, var_sum, robust_load

    def _compute_raw_objectives(self, assignment: List[int], tasks: List[dict],
                                servers: List[dict]) -> Tuple[float, float, float]:
        """计算原始目标值(未归一化)"""
        _, var_sum, robust_load = self._compute_server_loads(assignment, tasks, servers)

        f_j = np.array([s['f'] for s in servers])
        C_j = np.array([s['C'] for s in servers])

        # O1: Robust Makespan
        O1 = float(np.max(robust_load / f_j))

        # O2: Robust Load Imbalance (鲁棒负载标准差，与论文一致)
        O2 = float(np.std(robust_load))

        # O3: Server-level Uncertainty Risk (新版：服务器视角)
        # 公式：O3 = Σ_j (σ_j / max(Gap_j, ε))
        # 其中 σ_j = sqrt(var_sum_j), Gap_j = theta*C_j - robust_load_j
        gap_j = self.theta * C_j - robust_load
        epsilon = 0.01 * np.mean(C_j)
        std_j = np.sqrt(var_sum)
        O3 = sum(
            std_j[j] / max(gap_j[j], epsilon)
            for j in range(len(servers))
        )

        return O1, O2, O3

    def _init_normalization_bounds(self, population: List[List[int]],
                                   tasks: List[dict], servers: List[dict]):
        """基于当前种群确定归一化边界"""
        O1_vals, O2_vals, O3_vals = [], [], []

        for ind in population:
            O1, O2, O3 = self._compute_raw_objectives(ind, tasks, servers)
            O1_vals.append(O1)
            O2_vals.append(O2)
            O3_vals.append(O3)

        self.O1_min, self.O1_max = min(O1_vals), max(O1_vals)
        self.O2_min, self.O2_max = min(O2_vals), max(O2_vals)
        self.O3_min, self.O3_max = min(O3_vals), max(O3_vals)

    def _evaluate(self, assignment: List[int], tasks: List[dict],
                  servers: List[dict], g: int) -> float:
        """计算适应度(含动态惩罚)"""
        O1, O2, O3 = self._compute_raw_objectives(assignment, tasks, servers)

        O1_norm = (O1 - self.O1_min) / max(self.O1_max - self.O1_min, 1e-6)
        O2_norm = (O2 - self.O2_min) / max(self.O2_max - self.O2_min, 1e-6)
        O3_norm = (O3 - self.O3_min) / max(self.O3_max - self.O3_min, 1e-6)

        F = self.w1 * O1_norm + self.w2 * O2_norm + self.w3 * O3_norm

        _, _, robust_load = self._compute_server_loads(assignment, tasks, servers)
        C_j = np.array([s['C'] for s in servers])

        lambda_g = self.lambda_0 * (1 + self.beta * g)
        violation = np.sum(np.maximum(0, robust_load - self.theta * C_j) ** 2)
        penalty = lambda_g * violation

        return F + penalty

    def _tournament_selection(self, population: List[List[int]],
                              fitness: List[float]) -> List[List[int]]:
        """二元锦标赛选择"""
        parents = []
        n = len(population)
        for _ in range(n):
            i, j = np.random.randint(n, size=2)
            winner = population[i] if fitness[i] < fitness[j] else population[j]
            parents.append(winner.copy())
        return parents

    def _uniform_crossover(self, parents: List[List[int]]) -> List[List[int]]:
        """均匀交叉"""
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

    def _hybrid_mutation(self, individual: List[int], tasks: List[dict],
                         servers: List[dict]) -> List[int]:
        """混合变异: De-risking + Random"""
        if np.random.random() > self.p_m:
            return individual

        ind = individual.copy()
        n_servers = len(servers)

        if np.random.random() < self.p_risk:
            # De-risking Mutation
            _, _, robust_load = self._compute_server_loads(ind, tasks, servers)
            C_j = np.array([s['C'] for s in servers])
            gap_j = self.theta * C_j - robust_load

            epsilon = 0.01 * np.mean(C_j)
            risk_scores = []
            for i, j in enumerate(ind):
                delta_i = tasks[i]['mu'] + self.kappa * tasks[i]['sigma']
                score = delta_i / max(gap_j[j], epsilon)
                risk_scores.append((i, j, score))

            i_risk, j_src, _ = max(risk_scores, key=lambda x: x[2])
            j_tgt = int(np.argmax(gap_j))

            if gap_j[j_tgt] > gap_j[j_src] and j_tgt != j_src:
                ind[i_risk] = j_tgt
        else:
            # Random Mutation (Safe Mode)
            i = np.random.randint(len(ind))
            j_old = ind[i]

            # 计算当前Gap
            _, _, robust_load = self._compute_server_loads(ind, tasks, servers)
            C_j = np.array([s['C'] for s in servers])
            gap_j = self.theta * C_j - robust_load

            # 智能筛选: 只变异到有剩余空间(Gap>0)的服务器
            candidates = [j for j in range(n_servers) if j != j_old and gap_j[j] > 0]

            if candidates:
                j_new = candidates[np.random.randint(len(candidates))]
            else:
                # 如果都满了,选择溢出最少的(Gap最大)
                j_new = int(np.argmax(gap_j))

            ind[i] = j_new

        return ind

    def _elitist_replacement(self, population: List[List[int]], fitness: List[float],
                             offspring: List[List[int]], offspring_fitness: List[float]
                             ) -> Tuple[List[List[int]], List[float]]:
        """精英保留策略"""
        combined = list(zip(population + offspring, fitness + offspring_fitness))
        combined.sort(key=lambda x: x[1])

        new_pop = [x[0] for x in combined[:self.n_pop]]
        new_fit = [x[1] for x in combined[:self.n_pop]]

        return new_pop, new_fit

    def _get_best_feasible(self, population: List[List[int]], fitness: List[float],
                           tasks: List[dict], servers: List[dict]) -> List[int]:
        """返回最优可行解，无可行解时返回违规最小的解

        [修正] 使用L2范数衡量违规度，与_evaluate中的惩罚项保持一致
        """
        C_j = np.array([s['C'] for s in servers])

        best_feasible_ind = None
        best_feasible_fit = float('inf')
        min_violation_ind = None
        min_violation_val = float('inf')

        for ind, fit in zip(population, fitness):
            _, _, robust_load = self._compute_server_loads(ind, tasks, servers)
            # [修正] 使用L2范数（平方和），避免单点爆破
            violation = np.sum(np.maximum(0, robust_load - self.theta * C_j) ** 2)

            if violation < 1e-6:  # 可行解
                if fit < best_feasible_fit:
                    best_feasible_fit = fit
                    best_feasible_ind = ind
            else:  # 不可行解
                if violation < min_violation_val:
                    min_violation_val = violation
                    min_violation_ind = ind

        # 优先返回可行解，否则返回违规最小的解(安全保底)
        return best_feasible_ind if best_feasible_ind is not None else min_violation_ind
