#导入相关包
import numpy as np
import matplotlib.pyplot as plt
import math
#参数设定

number_of_bandits=10#老虎机的个数
number_of_arms=10#老虎机的手臂数
number_of_pulls=10000#尝试推老虎机的次数
epsilon=0.3 #Epsilon-Greedy算法参数
min_temp = 0.1 #Epsilon-Greedy参数
decay_rate=0.999 #Epsilon-Greedy参数


def pick_arm(q_values,counts,strategy,success,failure):
	#函数：随机返回一个臂

	global epsilon #全局变量
	#随机算法
	if strategy=="random":
		return np.random.randint(0,len(q_values))
		#随机对number_of_arms大小内，在范围内随机生成一个随机整数与一个老虎机对应

	#贪心算法，直接选择最大q_VAlues中收益最大的数字,只与统计收益q_values有关
	#贪心算法依赖于之前收益进行判定选择那一个
	if strategy=="greedy":
		best_arms_value = np.max(q_values)
		best_arms = np.argwhere(q_values==best_arms_value).flatten()
		#np.argwhere寻找q_VAlues对应最大价值的位置，因为q_VAlues返回非0的数组元组的索引
		return best_arms[np.random.randint(0,len(best_arms))] #再出现有多个q——VALues相同时，进行随机选择

	#最小量贪婪算法，加入一个epsilon参数，以1-epsilon的概率选取当前收益最⼤的臂，以epsilon的概率随机选取⼀个臂
	#
	if strategy=="egreedy" or strategy=="egreedy_decay": 
		if  strategy=="egreedy_decay": 
			epsilon=max(epsilon*decay_rate,min_temp)
			#egreedy_decay衰减，epsilon会随着时间增长减少，前期趋近于多做尝试，后期趋近于选择最大收益的那个
			#decay_rate,min_temp分别为衰减速率和最小下线

		#上述步骤时加入衰减的情况下先计算衰减,后进行判定
		#生成一个0-1的随机数字，作为概率判定，如随机大于epsilon选择当前收益最大的臂
		if np.random.random() > epsilon:
			#选择概率最大的臂
			best_arms_value = np.max(q_values)
			best_arms = np.argwhere(q_values==best_arms_value).flatten()
			return best_arms[np.random.randint(0,len(best_arms))]
		else:
			#随机选择臂
			return np.random.randint(0,len(q_values))

	# UpperConfidenceBound（UCB）算法，均值越⼤，标准差越⼩，被选中的概率会越来越⼤
	if strategy=="ucb":
		total_counts = np.sum(counts) #统计每一个臂加起来选择的次数
		q_values_ucb = q_values + np.sqrt(np.reciprocal(counts+0.001)*2*math.log(total_counts+1.0))
		#UCB指标进行概率判定，在当前收益上加一个开方数
		# 分目数np.reciprocal求倒数对数组的数字分别求导，出现次数+0.001，防止分母为0，
		# 代表对于一个臂来说，之前出现的次数越多，降低他的指标，减少其下次出现的概率
		# 分子数*math.log(total_counts+1.0)
		# 代表当进行拉臂尝试的次数越多，相当于增加每一个臂出现的概率
		# 也就是说判定选择臂的指标的UCB数值是两部分平衡得到。
		# 一方面是之前的收益，收益越高指标越高。
		# 另一方面是尝试的次数，经过尝试越多，趋向于将一些未出现的臂指标就会越来越高，出现多的臂指标降低
		best_arms_value = np.max(q_values_ucb)
		best_arms = np.argwhere(q_values_ucb==best_arms_value).flatten()
		return best_arms[np.random.randint(0,len(best_arms))]

	#汤普森算法
	if strategy=="thompson":
		sample_means = np.zeros(len(counts))#手臂的数量
		for i in range(len(counts)):#对每一个手臂
			sample_means[i]=np.random.beta(success[i]+1,failure[i]+1)
			#根据之前统计的对于第i个臂的成功和失败的结果，生成一个beta分布的概率随机值
		return np.argmax(sample_means)#返回最大概率值


fig = plt.figure()
ax = fig.add_subplot(111)
# for st in ["greedy","random","egreedy","egreedy_decay","ucb","thompson"]:
for st in ["thompson"]:
	#测试不同策略效果
	best_arm_counts = np.zeros((number_of_bandits,number_of_pulls))
	#生成counts统计数组，大小为老虎机*推臂次数

	#针对每一个老虎机的臂来说(其实作用是为了降低随机波动)
	#每一次都会随机生成每一个臂的随机概率收益矩阵（百分之多少的概率获得reward），多次重复测试
	for i in range(number_of_bandits):
	# for i in range(1):
		arm_means = np.random.rand(number_of_arms)#生成每一个臂的随机收益矩阵
		best_arm = np.argmax(arm_means) #找到其中最高收益对应臂序列号

		#初始化每一个臂的收益统计矩阵，设定为零
		q_values = np.zeros(number_of_arms) #将q值设定为零的臂数行
		counts = np.zeros(number_of_arms) #统计每一个臂选择到的次数
		success=np.zeros(number_of_arms) #统计成功的arms
		failure=np.zeros(number_of_arms)#统计失败的arms

		for j in range(number_of_pulls):
			#针对每一次推臂
			a = pick_arm(q_values,counts,st,success,failure)
			#随机生成的臂对应序列号，利用算法获得选择那一个臂
			#贪心算法，直接找到收益最大的索引
			reward = np.random.binomial(1,arm_means[a])
			#arm_means为每一个臂的收益矩阵（是随机生成的）
			#针对所选择的臂，其是否获得收益
			#numpy.random.RandomState.binomial(n, p, size=None)
			#个二项分布进行采样1,arm_means[a]对应n，p参数,size为采样次数,函数的返回值表示n中成功（success）的次数
			#相当于通过概率统计的方法，当随机推动第a个臂时，第a个臂获利的概率，进行二项分布统计输出结果

			counts[a]+=1.0 #随机到第a个臂在其counts中+1

			#更新第a个臂的收益
			#收益统计计算在第一次选到a臂时，其收益为1，在之后加入
			q_values[a]+= (reward-q_values[a])/counts[a]
			#总体目的是计算对于a臂统计，在进行前j次拉臂中q_values的概率收益
			#第a个臂 在前j次拉臂中，获得收益的次数处于抽到的总次数
			#对应贪心算法：在拉动收益最高的臂时，当设定多次实验时，最终结果不固定，依赖于随机生成的收益概率矩阵，如果实验足够多次，每一个

			#如果reward收益为1
			success[a]+=reward
			failure[a]+=(1-reward)
			#成功+1 失败+1

			best_arm_counts[i][j] = counts[best_arm]*100.0/(j+1)
			# 设定对于第i个老虎机的臂，对于第j次拉臂时，统计选择到最高收益的臂的个数处于当前拉臂次数
			# 相当于前j次拉臂后，最高收益选择的百分比
			#对于random随机，直接随机选取

		epsilon=0.3

	ys = np.mean(best_arm_counts,axis=0)#求平均值，转化为1*推臂次数
	# ys = best_arm_counts[0]
	xs = range(len(ys))
	ax.plot(xs, ys,label = st)

plt.xlabel('Steps')
plt.ylabel('Optimal pulls')

plt.tight_layout()
plt.legend()
plt.ylim((0,110))
plt.show()        