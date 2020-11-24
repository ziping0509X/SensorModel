# **说明**
这是一个使用深度强化学习反向拟合正弦函数的程序。主要思路是以输入的正弦函数sin（omega * t）作为环境状态集，以智能体的动作刚好抵消这个函数值作为奖励目标，反复训练这个智能体。

最终呈现的效果良好，智能体已经可以以百分之百的概率正确选择自己的动作。

文件中的figure1-3表现了算法结果，reward.csv和loss.csv则分别使用Excel记录了奖励函数和损失函数。
## 调参
**调参主要从以下几方面考虑**：

（1）Activation Function：需要根据输出的具体情况选择输出层的激活函数，为了能够包含正半轴和负半轴，这里需要选择tanh函数。

（2）learning rate：学习率控制了DQN在使用ADAM优化算法调参时的梯度策略，学习率越小，越容易收敛，但也越容易局限在局部最优解中；学习率越大，越容易发散，但也越容易找到全局最优解。

（3）reset reward：**奖励函数是DQN网络与优化目标链接的唯一桥梁**，需要根据具体情况设计。在这里需要注意的是，给模型的惩罚尺度不能太大，否则模型不容易学习到新的知识。

（4）batch size：初始模型选择的batch_size=256,训练效果并不好，将batch_size设定为128后好了很多。

（5）deep：由于这个模型的环境状态集和动作状态集都比较小，所以深度和神经元节点数可以适当减小一些。
## 图片
![](https://github.com/ziping0509X/SensorModel/blob/master/Figure_1.png)
![](https://github.com/ziping0509X/SensorModel/blob/master/Figure_2.png)
![](https://github.com/ziping0509X/SensorModel/blob/master/Figure_3.png)



