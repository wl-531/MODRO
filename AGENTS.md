# AGENTS.md — 实验代码指南（给 Codex 使用，要讲中文）

## 一、核心原则
- 态度：建设性协助；主动发现问题并提出方案；不确定就说“我不知道”。
- 风格：简单 > 复杂；可读 > 优雅；能跑 > 完美；实验说话 > 主观预测。

---

## 二、必须做（Hard Requirements）

### 实验规范
- 固定随机种子以确保可复现。
- 每 N 次保存中间结果，避免训练中断。
- 所有依赖库必须写明版本号。
- 在开始实验前给出说明：对比算法、参数依据、评估指标、资源占用估计。

### 参数示例（对 Codex 的参考写法）
```python
# learning_rate = 0.001  # Adam 推荐值，常用范围 1e-4 ~ 1e-2
# batch_size    = 32     # 取决于显存，常见 16/32/64
import random, numpy as np
random.seed(42)
np.random.seed(42)
```

三、禁止做（Strictly Forbidden）
• 自作主张添加未要求的功能。
• 过度抽象：ABC、Protocol、工厂模式、策略模式等。
• 过度模块化：能放一个文件，不要拆三个文件。
• 过度配置化：能硬编码的先硬编码，需要再参数化。
• 冗长的 docstring（一行说清楚即可）。
• 预测实验结果的具体数值。
• 在不确定时“不懂装懂”，不得猜测 API 行为。
￼
四、代码风格规范
❌ 不要这样（复杂类层次、抽象过度）
python
￼
复制代码
class AbstractScheduler(ABC
):
    @abstractmethod
    def schedule(self, tasks: List
[Task]) -> Assignment:
        """Schedule tasks to servers."""
        pass
✅ 要这样（短函数、简单直观）
python
￼
复制代码
def solve_batch(tasks, servers
):
    """求解一个批次的任务分配"""
    ...
其他要求
• 函数尽量短小（< 30 行）。
• 变量名应自解释，不要 cryptic abbreviations。
• 尽量使用 dataclass，而非复杂类体系结构。
• 直接 print 调试，不用 logging 系统（除非必须）。
• 每个实验一个脚本，开箱即跑。
￼
五、鼓励做（Encouraged）
• 质疑我的方案并提出替代建议。
• 指出性能瓶颈、内存风险、复杂度问题。
• 若不确定库用法，应主动提示：“我需要查文档”。
￼
六、研究背景（让 Codex 理解任务领域）
• 方向：边缘计算、任务卸载、鲁棒调度。
• 风格：务实，可行性 > 形式上的创新性。
• 时间：工作量可控，估算保守。
￼
七、Codex 的工作模式要求
• 修改代码前必须提供 Git 风格 patch，并等待明确批准。
• 不得一次性重构整个项目，只能进行“最小必要修改”。
• 在涉及数学模型（MODRO/ROSA）时，保持与论文公式一致。
