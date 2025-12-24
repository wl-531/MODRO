"""全局参数配置 - Final Heterogeneous Version"""

# 风险参数 (论文标准)
ALPHA = 0.15      # CVR目标阈值
KAPPA = 2.38      # 不确定性系数 (kappa = sqrt(1/alpha - 1))
THETA = 1.0       # 当前版本不再缩减容量，视为无额外折减

# ROSA 参数 (进化/鲁棒)
N_POP = 100        # 种群规模
G_MAX = 100       # 最大迭代次数
T_MAX = 1       # 最大运行时间(秒)
P_C = 0.8         # 交叉概率
P_M = 0.2         # 变异概率
P_RISK = 0.7      # De-risking变异概率
N_ELITE = 3       # 保护精英解数量
LAMBDA_0 = 5.0   # 初始惩罚系数
BETA = 0.3        # 惩罚系数增长速度

# 目标权重 (异构场景推荐)
W1 = 0.4         # Robust Makespan 权重
W2 = 0.25         # Robust Load Imbalance 权重
W3 = 0.35        # Marginal Uncertainty Risk 权重

# 实验物理参数（压力测试配置 - 调整至临界可行）
BATCH_SIZE = 800                # 略降批量，减轻鲁棒负载
N_SERVERS = 8                   # 服务器数量
DECISION_INTERVAL = 46         # 略增决策窗口，目标 rho_eff≈0.9-1.0

# 任务工作量参数
MU_RANGE = (50, 60)            # 期望工作量范围
CV_RANGE = (0.48, 0.62)         # 适度波动；bimodal 模式下此参数被忽略
#CV_RANGE = (0.20, 0.42)

# 蒙特卡洛
MC_SAMPLES = 10000              # 蒙特卡洛采样次数
