---
typora-root-url: ..\image 
---

#  Match Algorithms and Practices (Part 1)

**电商环境下的个性化推荐，主要包含三大维度的模块，针对用户的候选召回（Match），候选商品的精排（Rank），以及线上的策略调控（Re-rank）**

**。而Match（召回）和Rank（排序）是推荐流程非常关键的两步。**

![1543924311138](/../SecondWeek1202/1543924311138.png)

Match & Rank

定义：Match基于当前user(profile、history) 和context，快速在全库里找到To p N最相关的item，给Rank来做小范围综合多目标最大化。

通常做法：通常情况下，用各种算法做召回，比如user/item/model-based CF、Content-based、Demographic-based、DNN-Embedding-based等等，做粗排之后交由后面的Rank层做更精细的排序，最终展现TopK item。  

## 推荐系统的 Match 模块介绍

### （Match）对用户的候选召回

Match即有效和丰富的召回，从全量商品（还包括feed和视频等）集合中根据用户行为和属性信息触发尽可能多正确的结果，并将结果返回给Rank。

**推荐不同与搜索，在没有明确Query触达的情况下，用户的Match召回就显得尤为重要，希望相关性的信息尽可能的丰富。**

因此Match面对的是整个商品库集合，需要保留尽可能多的相关结果，剔除相关性较弱的结果，降低对后面链路的压力。

由此需综合利用多种信息，比如用户信息（年龄、性能、购买力等）、类目信息、店铺信息、文本信息等。从而既保证高召回率，又要保证高的准确率。Match结果的好坏，对整个结果有重要的影响。

Match 算法典型应用(针对不同情况采用不同的召回方法)：

* 猜你喜欢多样推荐

* 相似推荐看了还看

* 搭配推荐买了还买

**Collaborative Filtering **协同过滤

 协同过滤一般是在海量的用户中发掘出一小部分和你品位比较类似的，在协同过滤中，这些用户成为邻居，然后根据他们喜欢的其他东西组织成一个排序的目录作为推荐给你。当然其中有一个核心的问题：

-  如何确定一个用户是不是和你有相似的品位？
-  如何将邻居们的喜好组织成一个排序的目录？

协同过滤相对于集体智慧而言，它从一定程度上**保留了个体的特征**，就是你的品位偏好，所以它更多可以作为**个性化推荐** 的算法思想。可以想象，这种推荐策略在 Web 2.0 的长尾中是很重要的，将大众流行的东西推荐给长尾中的人怎么可能得到好的效果，这也回到推荐系统的一个核心问题：了解你的用户，然后才能给出更好的推荐。

要实现协同过滤，需要一下几个步骤

-  收集用户偏好
-  找到相似的用户或物品
-  计算推荐

数据矩阵化：

![1543924259976](/1543924259976.png)

### 基于共现关系的Collaborative Filtering 算法两类：

* User-based CF
  基本思想相当简单，基于用户对物品的偏好找到相邻邻居用户，然后将邻居用户喜欢的推荐给当前用户。计算上，就是将一个用户对所有物品的偏好作为一个向量来计算用户之间的相似度，找到
  K 邻居后，根据邻居的相似度权重以及他们对物品的偏好，预测当前用户没有偏好的未涉及物品，计算得到一个排序的物品列表作为推荐。图 2给出了一个例子，对于用户 A，根据用户的历史偏好，这里只计算得到一个邻居 - 用户 C，然后将用户 C 喜欢的物品 D 推荐给用户 A。

  ![1543924517050](/../SecondWeek1202/1543924517050.png) 

* item-based CF

基于物品的 CF 的原理和基于用户的 CF 类似，只是在计算邻居时采用物品本身，而不是从用户的角度，即基于用户对物品的偏好找到相似的物品，然后根据用户的历史偏好，推荐相似的物品给他。从计算的角度看，就是将所有用户对某个物品的偏好作为一个向量来计算物品之间的相似度，得到物品的相似物品后，根据用户历史的偏好预测当前用户还没有表示偏好的物品，计算得到一个排序的物品列表作为推荐。图 给出了一个例子，对于物品 A，根据所有用户的历史偏好，喜欢物品 A 的用户都喜欢物品 C，得出物品 A 和物品 C 比较相似，而用户C 喜欢物品 A，那么可以推断出用户 C 可能也喜欢物品 C。

![1543924651403](/../SecondWeek1202/1543924651403.png)

**对比**

![1543924682455](/../SecondWeek1202/1543924682455.png)

**ItemCF的推荐算法调用示意图 **  

![1543924772249](/../SecondWeek1202/1543924772249.png)

### Item CF 算法最新实践

一般的item相似计算
$$
w_i, j =\frac{|N(i)\cap N(j)|}{|N(i)|}
$$
N(i)N(j)分别表示喜欢物品i和物品j的用户数字

理解：喜欢物品i的用户有多少比例的用户也喜欢物品j

**改进版 I2I** 

motivation:

哈利波特效应（比方哈利波特多个人都会看，并不能说明两个人兴趣近似，推荐没有意义，对冷门物体采取同样行为更能说明兴趣相似）

上述公式中，如果j很热门，喜欢i的用户基本上都会看，其权重参数即为1

热门用户,用户行为缺乏考虑（使用热度比较高的用户会有很多无效行为）

solution(分别对应上述解决办法)：

**热门Item降权：**

![1543925867719](/../SecondWeek1202/1543925867719.png) 

i，j分别是喜欢物品i和物品j的人的集合，解决哈利波特效应。

对于i和j的相似度

分母：物品j喜欢的人数越多，其权重越低，惩罚对应物品的权重，减轻热门物品会有很多相似的可能性。

挖掘长尾信息，避免出现热门的推荐，提高alph，加大对j的惩罚，alpha越大，覆盖率会越高，平均热门程度会降低，降低流行度提高新颖性

**热门用户降权：** 

![1543925263064](/../SecondWeek1202/1543925263064.png) 

针对经过物品相似度修正alpha = 0.5时

分子为即喜欢i有喜欢j的用户的合集，由于存在一些活跃用户（例如开书店的人会买很多书），当这种两本书都买的人出现的越多，权重，就会降低

**综合考虑相似度计算（公式 来源未找到）：**

![1543926731029](/../SecondWeek1202/1543926731029.png) 



**实时I2I**（针对新品的推荐问题，实时增量）

![1543930599354](/../SecondWeek1202/1543930599354.png)



![1543929919782](/../SecondWeek1202/1543929919782.png)

**论文：（腾讯 2015) TencentRec: Real-time Stream Recommendation in Practice, SIGMOD’15** 

增量更新：物品对的相似度可以用过item Countp和itemCountq和pairCount的来进行，并i企鹅其均可以通过增量计算来更新

**Hybrid i2i——Learning to Rank（排序算法！！！）**

（针对的问题：无监督学习，无法刻画场景差异，就是你也不确定时相似推荐、猜你喜欢、买了还买何类推荐场景），采用机器学习训练的方法进行自动学习权重。

![1543930959628](/../SecondWeek1202/1543930959628.png)



机器学习模型：（排序算法！！！）

输入特征：

Trigger-item Relavance: i2i_score/favor2favor sim/text sim...

Item Feature: video_ctr、video_pv、video_comment、

Trigger Feature: trigger_ctr、topic_ctr

模型：

Loss：Pairwise Loss，同时优化CTR、LikeR、FavorR

Lambdamart/Neural Nets

**论文：**Hybrid i2i—— Learning to Rank for Information Retrieval Tie-Yan Liu

 适用于排序场景：不是传统的通过分类或者回归的方法求解排序问题，而是直接求解

原理大致认识：通过类似三元组损失的方法

https://blog.csdn.net/huagong_adu/article/details/40710305

### Model Based CF

由于基于邻域（user,item）的方法简单性和有效性，他们能够产生准确的个性化的推荐，而这个推荐往往是比较流行的产品。同时，还受到规模的限制，因为需要计算用户之间或商品之间的相似度，最差的情况时间复杂度为O(m*n)，一般情况下为O(m+n).在大规模数据下,用户的评分是个稀疏的,只有很少比例的用户或者商品有评分.比如,在Mendeley中,有百万级文章而用户可能只读过其中的几百篇.假设两个用户每人读100篇文章那么也只有0.0002的占比.
基于模型的协同过滤算法能一定程度上克服基于领域方法的限制，与基于领域的方法直接使用user-item评分来预测新的item的评分不同的是基于模型的方法利用评分学习出一个有效的模型，利用机器学习的方法对user-item交互关系进行学习得到一个模型，一般来说，基于模型的协同过滤方法是一种比较好的建立协同过滤的推荐系统方法。有很多机器学习方法可以用于建立模型，比如，贝叶斯网、聚类、分类、回归、矩阵分解、受限玻尔兹曼机等，有些方法在获得NetflixPrize的解决方案中用到过。 
**矩阵分解Matrix factorisation，如singular value decomposition(SVD), SVD++,将商品和用户分解到可以表示user-item隐性关系的空间中，矩阵因子的背后就是隐性特征如何去表示用户对商品的评分，这样我们就可以估计用户没有评分商品的评分。** 

![1543932942930](/../SecondWeek1202/1543932942930.png)

**Matrix Factorization (MF) 推荐算法** 

用latent vector来表示user和item(ID embedding)

组合关系用内积 inner product (衡量user对于某一类商品的偏好）

论文：Model Based CF 

论文：Deep Learning for Matching in Search and Recommendation

**为了优化L2loss，进行了SVD等系列方法的尝试**

**SVD：（https://www.cnblogs.com/LeftNotEasy/archive/2011/01/19/svd-and-applications.html）** 

![1543932633429](/../SecondWeek1202/1543932633429.png)SVD(singular value decomposition)，翻译成中文就是奇异值分解。SVD的用处有很多，比如：LSA（隐性语义分析）、推荐系统、特征压缩（或称数据降维）。SVD可以理解为：将一个比较复杂的矩阵用更小更简单的3个子矩阵的相乘来表示，这3个小矩阵描述了大矩阵重要的特性。

基于SVD的优势在于：用户的评分数据是稀疏矩阵，可以用SVD将原始数据映射到低维空间中，然后计算物品item之间的相似度，可以节省计算资源。

SVD is Suboptimal for CF 几个缺点

* Missing data和观测到的数据权重相同(>99% 稀疏性) 

* 没有正则项，容易过拟合

**改进SVD（Adjust SVD）**:

Factored Item Similarity Model 

MF 用UserID来表示用户（user-based）

另外一种做法是用用户评价过的item来表示用户（item-base）

![1543934719076](/../SecondWeek1202/1543934719076.png)

user表现/由用户u的评分过的项目/可以是当作item i和j之间的相似度来解释

**SVD++: Fusing User-based and Item-based CF**

MF (user-based CF) 用UserID来表示用户（直接映射ID到隐空间）

FISM (item-based CF) 用用户评价的item来表示用户（映射items到隐空间）

SVD++ 混合了两种想法



**(Side Info)融入更多的信息**

![1543935498189](/../SecondWeek1202/1543935498189.png)

综合多个属性进行推荐预测



**终极模型FM: Factorization Machines（分解机）**

FM 受到前面所有的分解模型的启发

每个特征都表示成embedding vector，并且构造二阶关系

FM 

允许更多的特征工程，并且可以表示之前所有模型为特殊的

FM（,MF,SVD++,timeSVD(Koren,KDD’09),PITF(Rendle,WSDM’10)etc.）

![1543935595599](/../SecondWeek1202/1543935595599.png)

**评分预测模型的loss分析**

![1543935693420](/../SecondWeek1202/1543935693420.png)

很多证据表明一个低MSE模型不一定代表排序效果好（MSE距离不一定是最好了，考虑余弦距离等）

 Possible Reasons:

* 均方误差(e.g., RMSE) and 排序指标之间的分歧

* 观察有偏用户总是去对喜欢的电影打分

### Towards Top-N Recommendation

论文：排序问题，Known as the Bayesian Personalized Ranking loss (Rendle, UAI’09).  优化相对序关系，而不是优化绝对值

## 实践：Movielens上的MF实战 (TensorFlow)

见.ipynb文件