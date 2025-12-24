"""服务器模型"""
from dataclasses import dataclass

@dataclass
class Server:
    """服务器类

    Attributes:
        f: CPU 频率(处理速度)
        C: 容量(决策周期内最大可承载工作量)
        L0: 已有负载(Pre-existing Load,确定值)
    """
    f: float
    C: float
    L0: float = 0.0

    def reset(self):
        """重置已有负载"""
        self.L0 = 0.0
