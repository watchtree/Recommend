

#  Industrial Recommendation System(工业推荐系统)——介绍与评估



![1543390353793](C:\Users\tree\AppData\Roaming\Typora\typora-user-images\1543390353793.png)

马太效应（Matthew Effect），指强者愈强、弱者愈弱的现象

[长尾](https://baike.baidu.com/item/%E9%95%BF%E5%B0%BE)效应，英文名称Long Tail Effect。“头”（head）和“尾”（tail）是两个统计学名词。正态曲线中间的突起部分叫“头”；两边相对平缓的部分叫“尾”。从人们需求的角度来看，大多数的需求会集中在头部，而这部分我们可以称之为流行，而分布在尾部的需求是个性化的，零散的小量的需求。而这部分差异化的、少量的需求会在[需求曲线](https://baike.baidu.com/item/%E9%9C%80%E6%B1%82%E6%9B%B2%E7%BA%BF/3351682)上面形成一条长长的“尾巴”，而所谓长尾效应就在于它的数量上，将所有非流行的市场累加起来就会形成一个比流行市场还大的市场。

推荐系统存在的前提：

* 信息过载
* ⽤户需求不明确

推荐系统的⽬标：

* ⾼效连接⽤户和物品，发现长尾商品（让小众生产者获得流量）
* 留住⽤户和内容⽣产者，实现商业⽬标

## 推荐系统评估

![1543390585163](C:\Users\tree\AppData\Roaming\Typora\typora-user-images\1543390585163.png)

反馈

![1543390606508](C:\Users\tree\AppData\Roaming\Typora\typora-user-images\1543390606508.png)

准确性：

一是用与评分预测，均方差根和均方误差

二是准确率和召回率（R为根据训练集上的行为给用于提供的推荐列表T为用户在测试集上的行为列表）

准确率代表用户在点击推荐列表中的东西占据其点击的百分数

召回率代表给用户推荐的东西被点击的部分占推荐整体的数量

![1543391837485](C:\Users\tree\AppData\Roaming\Typora\typora-user-images\1543391837485.png)

覆盖度：

![1543392241069](C:\Users\tree\AppData\Roaming\Typora\typora-user-images\1543392241069.png)

描述一个推荐系统对物品长尾的发掘能力

只用全面来作为覆盖内容的代表过于简单

因此细化评价指标

![1543392397590](C:\Users\tree\AppData\Roaming\Typora\typora-user-images\1543392397590.png)

![1543392857401](C:\Users\tree\AppData\Roaming\Typora\typora-user-images\1543392857401.png)

p(i)代表物品的流行度，为其数量占总数的百分比（通过统计推荐列表中不同物体出现次数的分布确定）





![1543392842495](C:\Users\tree\AppData\Roaming\Typora\typora-user-images\1543392842495.png)

Ij是按照物品流行度p从小到大排序的物品列表中的第j个物品，计算物品流行度分布

信息熵和基尼指数度量样本集合纯度

**基尼指数是信息熵中﹣logP在P=1处一阶泰勒展开后的结果！所以两者都可以用来度量数据集的纯度**

（https://blog.csdn.net/YE1215172385/article/details/79470926）

![1543393623463](C:\Users\tree\AppData\Roaming\Typora\typora-user-images\1543393623463.png)



推荐系统的初衷是消除马太效应，使得各种物品都能展示给对他们感兴趣的人群

使用基尼指数，如果从推荐列表中计算出物品流行度的基尼系数大于初始用户行为中计算出的物品流行度的基尼系数。

多样性&新颖性&惊喜性

多样性：推荐列表中两两物品的不相似性。（相似性如何度量？
新颖性：未曾关注的类别、作者；推荐结果的平均流⾏度
惊喜性：历史不相似（惊）但很满意（喜）

往往需要牺牲准确性
使⽤历史⾏为预测⽤户对某个物品的喜爱程度
系统过度强调实时性

Exploitation：选择现在可能最佳的⽅案（当前最优解）
 Exploration：选择现在不确定的⼀些⽅案，但未来可能会有⾼收益（扩展、探索新的内容）
的⽅案。
在做两类决策的过程中，不断更新对所有决策的不确定性的认知，优化
长期的⽬标函数

用于平衡探索与当前最优的算法 bandit 算法

以下列参数做说明：

```python
number_of_bandits=10#老虎机的个数
number_of_arms=10#老虎机的手臂数
number_of_pulls=10000#尝试推老虎机的次数
epsilon=0.3 #Epsilon-Greedy算法参数
min_temp = 0.1 #Epsilon-Greedy参数
decay_rate=0.999 #Epsilon-Greedy参数
```

老虎机的个数是用来对每一个算法进行多次尝试，降低波动的。

每一个老虎机有10个臂

推动10000次，以多次实验中取到最佳收益点的次数/推动总数作为算法评估的标准。

 bandit 算法（利用随机方法，或者在前面的实验中获得最高效益的方法，将保持和探索结合的策略）：

随机random算法：直接随机选用臂，在达到很大次数时，取到最大收益的概率的臂近似为1/10

Epsilon-Greedy小量贪婪算法：以1-epsilon的概率选取当前收益最⼤的臂（依赖于之前测试的结果），以epsilon的概率随机选取⼀个臂

Epsilon-Greedy-Decay:使得前期趋近于多做尝试，后期趋近于选择最大收益的那个，使得系统前期去探索更多可能。

![1543468062223](C:\Users\tree\AppData\Roaming\Typora\typora-user-images\1543468062223.png)、

Upper Confidence Bound（UCB）：均值越⼤，标准差越⼩，被选中的概率会越来越⼤![1543403184046](C:\Users\tree\PycharmProjects\CL_classify\%5CUsers%5Ctree%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5C1543403184046.png)

```python
#UCB指标进行概率判定，在当前收益上加一个开方数
# 分目数np.reciprocal求倒数对数组的数字分别求导，出现次数+0.001，防止分母为0，
# 代表对于一个臂来说，之前出现的次数越多，降低他的指标，减少其下次出现的概率
# 分子数*math.log(total_counts+1.0)
# 代表当进行拉臂尝试的次数越多，相当于增加每一个臂出现的概率
# 也就是说判定选择臂的指标的UCB数值是两部分平衡得到。
# 一方面是之前的收益，收益越高指标越高。
# 另一方面是尝试的次数，经过尝试越多，趋向于将一些未出现的臂指标就会越来越高，出现多的臂指标降低
```

Thompson Sampling：每个臂维护⼀个beta(wins, lose)分布，每次⽤现有的beta分布产⽣⼀个随机数，选择随机数最⼤的臂

![1543469456664](C:\Users\tree\PycharmProjects\CL_classify\image\%5CUsers%5Ctree%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5C1543469456664.png)



![1543469679804](C:\Users\tree\PycharmProjects\CL_classify\image\%5CUsers%5Ctree%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5C1543469679804.png)



Multi-armed bandit问题（多臂老虎机问题）（相当于在推荐系统中，探索部分和保持部分需要的比例）：**



在概率论中，多臂赌博机问题（有时也称为K臂/N臂赌博机问题），是一个赌徒需要在一排老虎机前决定拉动哪一个老虎机的臂，并且决定每个臂需要被拉动多少次的问题。每台老虎机提供的奖励是与它自身的奖励随机分布的函数相关的。赌徒的目标是最大限度地通过杠杆拉动序列，使得获得的奖励最大化

因此，在解决这个问题的时候，需要在“exploration”（探索新臂以获得跟多关于臂的回报的信息）和“exploitation”（选择已有回报最高的臂来获取最大利益）之中进行权衡

编程python实现见bandits.py

## 应用说明

Bandit算法-应⽤

• 兴趣探索
• 冷启动探索（对于新人的推荐）
• LinUCB：加⼊特征信息。⽤User和Item的特征预估回报及其置信区间，选择置信区间上界最⼤的Item推荐，观察回报后更新线性关系的参数，以此达到试验学习的⽬的。
• COFIBA：bandit结合协同过滤
• 基于⽤户聚类挑选最佳的Item（相似⽤户集体决策的Bandit）；
• 基于⽤户的反馈情况调整User和Item的聚类（协同过滤部分）

EE实践：

• 兴趣扩展：相似话题，搭配推荐
• ⼈群算法：userCF、⽤户聚类
• Bandit算法
• graph walking
• 平衡个性化推荐和热门推荐⽐例
• 随机丢弃⽤户⾏为历史
• 随机扰动模型参数

评估⽅法
• 问卷调查：成本⾼
• 离线评估:
• 只能在⽤户看到过的候选集上做评估，且跟线上真实效果存在偏差
• 只能评估少数指标，
• 速度快，不损害⽤户体验
• 在线评估：A/B testing
实践：离线评估和在线评估相结合，定期做问卷调查

## AB testing方法

单层实验：以某种分流的⽅法（随机、uid%100），给每个实验组
分配⼀定的流量。每个实验组配置不同的实验参数。
![1543470269871](C:\Users\tree\PycharmProjects\CL_classify\image\%5CUsers%5Ctree%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5C1543470269871.png)
• 只能⽀持少量实验，不利于迭代
• 实验之间不独⽴，策略之间可能相互影响
• 分流⽅式不灵活

多层重叠实验框架：

• 保留单层实验框架易⽤，快速的优点的同时，增加可扩展性，灵活性，健壮性。
• 核⼼思路：将参数划分到N个⼦集，每个⼦集都关联⼀个实验层，每个请求会被N个实验处理，同⼀个参数不能出现在多个层中。

![1543470306336](C:\Users\tree\PycharmProjects\CL_classify\image\%5CUsers%5Ctree%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5C1543470306336.png)

• 分配函数（流量在每层被打散的⽅法）如何设计？如何保证每层流量分配的均匀性和正交性？
• 如何处理实验样本的过滤（eg只选取某个地区的⽤户、只选取新⽤户）？
• 分配多⼤的流量可以使实验置信？

相关论文（非重点）《Overlapping Experiment Infrastructure:More, Better, Faster Experimentation》Google@KDD2010

![1543470424317](C:\Users\tree\PycharmProjects\CL_classify\image\%5CUsers%5Ctree%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5C1543470424317.png)

