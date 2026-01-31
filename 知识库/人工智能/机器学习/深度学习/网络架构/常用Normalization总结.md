![[Pasted image 20240309205836.png]]
常用的Normalization方法主要有：Batch Normalization（BN，2015年）、Layer Normalization（LN，2016年）、Instance Normalization（IN，2017年）、Group Normalization（GN，2018年）。它们都是从激活函数的输入来考虑、做文章的，以不同的方式**对激活函数的输入进行 Norm** 的。
将输入的 **feature map shape** 记为$[N, C, H, W]$，其中$N$表示batch size，即$N$个样本；$C$表示通道数；$H$、$W$分别表示特征图的高度、宽度。这几个方法主要的区别就是在：

1. BN是在batch上，对N、H、W做归一化，而保留通道 C 的维度。BN对较小的batch size效果不好。BN适用于固定深度的前向神经网络，如CNN，不适用于RNN；

2. LN在通道方向上，对C、H、W归一化，主要对RNN效果明显；

3. IN在图像像素上，对H、W做归一化，用在风格化迁移；

4. GN将channel分组，然后再做归一化。

# 现象
## Internal Covariate Shift
深度学习的训练过程可以看成很多层的叠加，而每一层的参数更新会导致下一层输入数据的分布发生变化，通过层层累加，高层的输入分布变化会非常剧烈导致上层的数据需要不断去变化以适应底层参数的更新。因此学习率，初始化权重等超参数的设置对模型的收敛非常重要，从而导致训练困难。总结下来，每个神经元的输入数据不再是独立同分布的，这样会造成：
1. 上层参数需要不断适应新的输入数据分布，降低学习速度
2. 下层输入的变化可能趋向于变大或者变小，导致上层落入饱和区，使学习过早停止
3. 每层的更新都会影响到其他层
![[Pasted image 20240220010655.png]]
在数据喂给机器学习模型之前，需要进行白化处理，去除特征之间的相关且保证所有的特征具有相同的均值和方差。
# Normalization 方法
基本思想是在将$x$送给神经元之前，先做平移和伸缩变换，将其分布规范在固定区间的标准分布：
$$
h=f(g\times\frac{x-u}{\sigma}+b)
$$
首先将输入向量$x$归一化为$(0,1)$的标注分布，其中$u$是均值，$\sigma$是方差。引入两个可学习参数$g$和$b$来保证网络的学习能力。

## 有效性分析
### 权重伸缩不变性
$$
\text{Norm}(\lambda WX)=\text{Norm}(g\frac{\lambda W X- \lambda u}{\lambda \sigma}+b)=\text{Norm}(WX)
$$
权重伸缩变化对反向传播没有影响。
### 数据伸缩不变性
$$
\text{Norm}(W\lambda X)=\text{Norm}(WX)
$$
**数据伸缩不变性仅仅对BN，LN成立，** 因为这两者在对数据进行规范化时，数据进行伸缩其均值和方差也会变化，分子分母相互抵消。
## Batch Normalization (BN)
### 为什么要进行BN呢？
1）在深度神经网络训练的过程中，通常以输入网络的每一个mini-batch进行训练，这样每个batch具有不同的分布，使模型训练起来特别困难。

（2）Internal Covariate Shift (ICS) 问题：在训练的过程中，激活函数会改变各层数据的分布，随着网络的加深，这种改变（差异）会越来越大，使模型训练起来特别困难，收敛速度很慢，会出现梯度消失的问题。
### BN原理

如图所示，一个batch共有$N$个shape为$C \times H \times W$的feature maps。

计算BN时，将沿维度$C$在维度$NHW$上操作。维度$C$上的所有数据记为$X_c$，如图蓝色部分。

batch内各通道的均值$\mu(X_c)$：
$$
\mu(X_c)=\frac{1}{NHW}\sum^N_{n=1}\sum^H_{h=1}\sum^W_{w=1}x_{nchw}
$$
batch内各通道的方差$\sigma^2(X_c)$：
$$
\sigma^2(X_c)=\frac{1}{NHW}\sum^N_{n=1}\sum^H_{h=1}\sum^W_{w=1}(x_{nchw}-\mu(X_c))^2
$$

经过standardization的值$\hat{x}$(满足均值为0方差为1的标准正态分布，$\epsilon$为防止除零引入的可忽略极小值):
$$
\hat{x}=\frac{x-\mu(X_c)}{\sqrt{\sigma^2(X_c)+\epsilon}}
$$
经过scale和shift得到的新分布：
$$
BN_{\gamma,\beta}(x)=\gamma\hat{x}+\beta
$$
经过BN后，输入batch的均值和方法将分别变为待学习的$\gamma$和$\beta$。

### BN优缺点和适用场景

#### 优点

* 由于分布标准化，可**使用更大的学习率，使训练过程更稳定，加快模型训练的收敛速度**。
* 因为 BN 的 Standardization 过程会移除直流分量/常量偏置，故可**令 bias=0**。
* **对权重初始化不再敏感**。通常初始权重采样自零均值某方差的高斯分布，以往对高斯分布的方差设置十分重要。有了 BN 后，对与其同一个输出节点相连的权重进行放缩，其标准差也会放缩同样倍数，最后相除抵消影响。同理，**对权重的尺度也不再敏感**。
* **深层网络可使用 Sigmoid 和 Tanh 了**，BN 使数据 跳出激活函数饱和区，抑制了梯度消失/弥散问题。
* BN 具有 某种正则作用，无需太依赖 Dropout 来缓解过拟合。

#### 缺点

* BN仅适用于通道数固定的场景，如CNN。对于RNN，LSTM和Transformer等序列长度不一致的网络而言，BN并不适用，通常改用LN。由于通道数不一致，会导致部分通道计算时batchsize变小，效果较差。

* BN在batch size值较小时难以通过少量样本来估计训练数据整体的均值和方差，效果较差。

#### 适用场景

CNN等通道数固定的网络
BN适用于每个mini-batch比较大，数据比较接近的场景，如CNN。然而而Batch Size过大又会占用过多的显存，一种折中的办法就是梯度累积来实现大Batch Size 的效果。

### BN细节

#### 卷积层的BN参数有多少？

1个kernel对应于1对$\gamma$和$\beta$参数。

#### BN可以没有scale和shift吗？

## LayerNorm (LN)
适用于batch内实例长度不定的问题。
## Instance Norm (IN)
Instance Norm在图像像素上对 H, W做归一化，相当于把每个通道的所有像素加起来，再除以该通道的像素总数。
主要用在图像的风格迁移。因为在图像风格化中，生成结果主要依赖于某个图像实例，feature map 的各个 channel 的均值和方差会影响到最终生成图像的风格。所以对整个batch归一化不适合图像风格化中，因而对H、W做归一化。可以加速模型收敛，并且保持每个图像实例之间的独立。

## Group Norm (GN)
Group Normalization (GN) 是针对BN在batch size较小时错误率较高而提出的改进算法。因为BN层的计算结果依赖当前batch的数据，当batch size较小时（比如2、4），该batch数据的均值和方差的代表性较差，因为对最后的结果影响较大。
GN将通道C分成了G份，每份 C//G，当G=1时，每份C个，即为一整个LN。
GN 适用于占用显存比较大的任务，例如图像分割。就像在显存不够的情况下，计算n个mini batch的梯度值后再进行梯度更新，实现小batch size更准确的梯度值。

# 参考资料
[最全Normalization](https://mp.weixin.qq.com/s/KS5x1B8gbfLLUAL6aHICrw)
[模型优化之Batch Normalization - 知乎](https://zhuanlan.zhihu.com/p/54171297)
[BatchNorm的原理及代码实现 - 知乎](https://zhuanlan.zhihu.com/p/88347589)

