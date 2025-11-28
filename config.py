"""全局参数配置"""

# 风险参数
ALPHA = 0.15  # CVR目标阈值
KAPPA = 2.38  # 不确定性系数 κ = √(1/α - 1), α=0.15时约2.38
THETA = 0.90  # 容量预留系数,从0.85放宽到0.90增加缓冲

# ROSA 参数
N_POP = 50      # 种群规模
G_MAX = 100     # 最大迭代次数
T_MAX = 0.1     # 最大运行时间(秒)
P_C = 0.8       # 交叉概率
P_M = 0.1       # 变异概率
P_RISK = 0.7    # De-risking变异概率
N_ELITE = 2     # 精英个体数量
LAMBDA_0 = 1.0  # 初始惩罚系数
BETA = 0.1      # 惩罚系数增长率

# 目标权重
W1 = 0.5  # Robust Makespan权重
W2 = 0.3  # Robust Load Imbalance权重
W3 = 0.2  # Marginal Uncertainty Risk权重

# 实验物理参数
BATCH_SIZE = 80                 # 每批任务数量
N_SERVERS = 5                   # 服务器数量
DECISION_INTERVAL = 8.5         # 决策周期(秒),黄金点,预期ρ_eff≈0.85,为ROSA提供优化空间

# 任务工作量参数
MU_RANGE = (10, 100)            # 期望工作量范围
CV_RANGE = (0.4, 0.6)           # 变异系数范围,从(0.3,0.5)改为(0.4,0.6),增加波动性

# 蒙特卡洛
MC_SAMPLES = 10000              # 蒙特卡洛采样次数
