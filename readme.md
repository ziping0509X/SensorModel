# **说明**
这是一个使用深度强化学习反向拟合正弦函数的程序。主要思路是以输入的正弦函数sin（omega * t）作为环境状态集，以智能体的动作刚好抵消这个函数值作为奖励目标，反复训练这个智能体。

最终呈现的效果良好，智能体已经可以以百分之百的概率正确选择自己的动作。
##调参
**调参主要从以下几方面考虑**：

（1）激活函数，需要根据输出的具体情况选择输出层的激活函数，为了能够包含正半轴和负半轴，这里需要选择tanh函数。
（2）learning rate
（3）reset reward
（4）
