"""全局参数配置"""

# 风险参数
ALPHA = 0.15      # CVR目标阈值
KAPPA = 2.38      # 不确定性系数 sqrt(1/alpha - 1)

# RA-LNS 参数
RA_LNS_PATIENCE = 15        # Stage-1A 不改进次数阈值
RA_LNS_DESTROY_K = 3        # Stage-1B destroy 任务数
RA_LNS_T_MAX = 0.01         # 10ms 时间预算
RA_LNS_EPS_DIV = 1e-6       # 除零保护：max(Gap, eps_div), max(sigma, eps_div)
RA_LNS_TOL_FEAS = 1e-9      # 可行性判断：Gap >= -tol_feas

# 实验参数
BATCH_SIZE = 425            # 任务批次大小
N_SERVERS = 10              # 服务器数量
DECISION_INTERVAL = 30      # 决策周期（秒）

# 任务工作量参数
MU_RANGE = (50, 60)         # 期望工作量范围
CV_RANGE = (0.48, 0.62)     # 变异系数范围

# 蒙特卡洛
MC_SAMPLES = 10000          # 蒙特卡洛采样次数
