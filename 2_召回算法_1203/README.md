---
typora-copy-images-to: image
typora-root-url: image
---

# Advanced Match Algorithm and Online Serving

目录：

embedding学习常⽤算法

* Matrix Factorization

* topic model(主题模型，文本分类信息检索，对应样本特征)

  生成模型PRSV 生成模型LDA

* word2vec（原理、代码、Airbnb应⽤）（词向量）（无监督学习）

* DNN（Youtube应⽤）

Online Match Serving

* 通⽤召回框架
* K-Nearest Neighbors search（LSH、Kd tree 、ball tree）

## embedding 

### embedding from MF

![1544346078755](C:\Users\tree\PycharmProjects\recommendation\2_召回算法_1203\image\1544346078755.png)

### embedding from topic model

![1544346106354](C:\Users\tree\PycharmProjects\recommendation\2_召回算法_1203\image\1544346106354.png)

### embedding from topic model

![1544346134776](C:\Users\tree\PycharmProjects\recommendation\2_召回算法_1203\image\1544346134776.png)

LDA in music recommendation：

建模：

* 歌曲(doc)-歌词(word)
* ⽤户(doc)-歌曲(word)

应⽤：

* 相似歌曲：根据doc的topic分布计算相似度
* ⽣成歌单：每个topic下概率最⼤的doc

特点：

不考虑顺序

出现对于低频词效果不好

### word2vec-由来

![1544346269368](C:\Users\tree\PycharmProjects\recommendation\2_召回算法_1203\image\1544346269368.png)

### word2vec-实现⽅法

![1544348315970](C:\Users\tree\PycharmProjects\recommendation\2_召回算法_1203\image\1544348315970.png)

* Skip-Ngram是根据word来预测context的概率P(context|word)

* CBOW(Continuous Bag of Words)：根据context来预测word概
  率P(word|context)

### word2vec example

![1544348444620](C:\Users\tree\PycharmProjects\recommendation\2_召回算法_1203\image\1544348444620.png)

### word2vec训练

中间的激活函数是softmax多分类概率，相当于有一个正例其他都是负例

一个词对应很多预估值，会慢

解决样本不平衡的问题

1. Hierarchical Softmax
   使⽤⼀颗⼆分Huffman树的表⽰，叶⼦结点是单词，词频越⾼离跟节点越低，优化计算效率O(logV)

2. Negative Sampling

![1544348876832](C:\Users\tree\PycharmProjects\recommendation\2_召回算法_1203\image\1544348876832.png)

小数据集多采负类，大数据集少采集负类

### 词向量的优势

•不丢失信息的情况下降低维度
•矩阵及向量运算便于并⾏
•向量空间具有物理意义
•可以在多个不同的维度上具有相似性
•线性规则：king - man = queen– woman

![1544426389519](C:\Users\tree\PycharmProjects\recommendation\2_召回算法_1203\image\1544426389519.png)

![1544426397054](C:\Users\tree\PycharmProjects\recommendation\2_召回算法_1203\image\1544426397054.png)

### word2vec at Airbnb

•详情页相似推荐：当前listing下的most relevant listings
•Click session=exploratorysessions + booked sessions，session切分依据？

通常可以以时间session或者地点session作为click分割

•book listing作为global context参与学习，提⾼转化率
•同⼀market下负采样（同一个地区）
•冷启动如何做？近邻embedding平均（新房间的冷启动 ）

![1544426552157](C:\Users\tree\PycharmProjects\recommendation\2_召回算法_1203\image\1544426552157.png)

离线评估

聚类观察：

按地点聚类

按建筑风格相似

房间类型相似（相同房价，相同房形相同）

在线评估：

房屋购买 

![1544427004412](C:\Users\tree\PycharmProjects\recommendation\2_召回算法_1203\image\1544427004412.png)

### embedding from DNN

![1544429708290](C:\Users\tree\PycharmProjects\recommendation\2_召回算法_1203\image\1544429708290.png)

### DNN at Google

training：通过分类任务学习出⽤户向量和视频向量： 

• 每个视频作为⼀个类别，观看完成是视频作为正例
• 把最后⼀层隐层输出作为⽤户向量（U+C）
• 视频向量？ 

* pre trained to feed 
*  training together

serving：输⼊⽤户向量，查询出与之向量相似度TopK⾼的视频

![1544429855871](C:\Users\tree\PycharmProjects\recommendation\2_召回算法_1203\image\1544429855871.png)

 ### DNN at Google

• 随机负采样效果好于hierarchical soft-max

1. 对于常见的单词对或者短语，在模型中将他们视为单个的单词。
2. 对常见单词进行二次采样来减少他们在训练样本中的数量。
3. 使用所谓的“负采样”（negative sampling）来改进优化对象，这将造成每一个训练的样本只会更对模型权重的很小一个比例的更新。

• 使⽤全⽹数据⽽不是只使⽤推荐数据
• 每个⽤户⽣成固定数量的样本（部分用户看的过多视频影响）
• 丢弃搜索词的序列性（不考虑时序，解决推荐密集的问题）
• 输⼊数据只使⽤历史信息 

![1544438973302](C:\Users\tree\PycharmProjects\recommendation\2_召回算法_1203\image\1544438973302.png)

### how to choose algorithm？

• 有监督 or ⽆监督？（用户收藏列表，word2vec，LDA无监督，用户点击序列有点堵）
• 有序 or ⽆序？（看文字顺序无所谓，看视频有顺序）
• item量级（高频词汇（不能用LDA））
• 实时性(NN实时方便，LDA不方便)
• 多样性
• 业务场景⽬标

## Online Match Serving

### match online serving

![1544453281116](C:\Users\tree\PycharmProjects\recommendation\2_召回算法_1203\image\1544453281116.png)

### key-value storage

•NoSQL存储系统：存储键值对⽂档，修改灵活；⽆JOIN操作，操作简单，速度快
•kv存储是NoSQL存储的⼀种

* hbase：分布式、持久化，常⽤于⼤数据存储
* redis：基于内存、速度快，常⽤于缓存

### match online serving 

![1544495221562](C:\Users\tree\PycharmProjects\recommendation\2_召回算法_1203\image\1544495221562.png)

参数存储

### match online serving

•如何选取key？（⽤户偏好建模）
•如何查询value？（K -Nearest Neighbors search）

### Sharding（用于优化大数据的近邻搜索）

![1544495569676](C:\Users\tree\PycharmProjects\recommendation\2_召回算法_1203\image\1544495569676.png)

### Hashing（用于优化大数据的近邻搜索）

![1544495656134](C:\Users\tree\PycharmProjects\recommendation\2_召回算法_1203\image\1544495656134.png)

近似搜索（方便实现且维数增大不受影响）

局部敏感哈希：

### k-d tree-定义（用于优化大数据的近邻搜索）

聚类，只算近似度较高的类

• k-dimension tree，k维欧⼏⾥德空间组织点的数据结构
• 每个节点代表⼀个超平⾯，该超平⾯垂直于当前划分维度的坐标轴
• ⼆叉树，每个节点分裂时选取⼀个维度。其左⼦树上所有点在d维的坐标值均⼩于当前值，右⼦树上所有点在d维的坐标值均⼤于等于当前值
• 平衡的k-d tree所有叶⼦节点到根节点的距离近似相等

### k-d tree-构建

1. 随着树的深度轮流选择轴当作分割⾯。
2. 取数据点在该维度的中值作为切分超平⾯，将中值左侧的数据点挂在其左⼦树，将中值右侧的数据点挂在其右⼦树。
3. 递归处理其⼦树，直⾄所有数据点挂载完毕（叶⼦节点可以包含多个数据）

![1544496699296](C:\Users\tree\PycharmProjects\recommendation\2_召回算法_1203\image\1544496699296.png)

### k-d tree-最近邻搜索

在建好的k-d tree上搜索(3,5)的最近邻
1. 从根节点开始，递归的往下移

2. 移动到叶节点，并当作”⽬前最佳点”

3. 解开递归，并对每个经过的节点运⾏下列步骤
    • 如果⽬前所在点⽐⽬前最佳点更靠近输⼊点，则将其变为⽬前最佳点
    • 检查另⼀边⼦树有没有更近的点，如果有则从该节点往下找

4. 当根节点搜索完毕后完成最邻近搜索

   ![1544496937933](C:\Users\tree\PycharmProjects\recommendation\2_召回算法_1203\image\1544496937933.png)

   ### ball tree-定义

   可剪枝

   论文：Ball*-tree: Efficient spatial indexing for constrained nearest-neighborsearch in metric spaces（对Ball*-tree的优化）

   相似点在同一个地方

![1544497340805](C:\Users\tree\PycharmProjects\recommendation\2_召回算法_1203\image\1544497340805.png)

![1544497359652](C:\Users\tree\PycharmProjects\recommendation\2_召回算法_1203\image\1544497359652.png)



### ball tree-构建

• 初始化⼀个根节点
• 选择离根节点最远的节点A作为左⼦树的中⼼节点
• 选择离A最远的节点B作为左⼦树的中⼼节点
• 将其它节点分配到⾥中⼼节点最近的⼦树，递归建树

![1544498279423](C:\Users\tree\PycharmProjects\recommendation\2_召回算法_1203\image\1544498279423.png)

### ball tree-最近邻搜索

剪枝

• Ds：⽬标点与当前最优的k个点的最⼤距离
• Dn:当前点n和⽬标点t的距离：Dn = max{Dn.Parent, |t −center(N)|（当前结点减去中心结点 ） − radius(N)（半径）} 

• 当Dn < Ds时继续搜索
• 叶⼦节点中⼩于Ds的点加⼊队列，并更新Ds。队列⼤于k时移出最远的点

![1544498634831](C:\Users\tree\PycharmProjects\recommendation\2_召回算法_1203\image\1544498634831.png)、】 

分解搜索

例如对于1E的候选集，先搜索100W的中心结点，再继续中心结点对用的100维里搜索

### K -Nearest Neighbors search

• offline computation + online storage（离线计算+在线实时更新）
• Brute force（暴力计算）
• sharding（并⾏Brute force）
• Hashing（近似结果key-value）
• space partition（ball tree常用）（近似结果）
• 只计算部分类别下的距离（近似结果）

召回率评估