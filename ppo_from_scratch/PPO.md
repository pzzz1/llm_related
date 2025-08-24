# PPO数学解析
## 基础概念
### 4+2理解法
四个模型     
策略模型：待优化的模型，参与参数更新  
奖励模型：计算当前动作的即时奖励，不参与参数更新  
价值模型：计算当前动作和状态的期望回报，可由奖励模型或者策略模型初始化得到，参与参数更新  
参考模型：由策略模型初始化得到，不参与参数更新，用于限制策略模型在优化时不偏离原始模型太远  
两个损失  
策略损失：用于优化策略模型  
价值损失：用于优化价值模型  
### 一些基本概念
策略policy： 输入State，得到Action的概率分布，一般表示为π，RLHF中，策略表示的就是我们需要优化的大模型，从策略进行采样的过程就是大模型生成句子的过程  
状态state： 采样得到的当前的状态，可以理解为大模型目前已经生成的句子。  
动作Action: 模型做出的动作，即大模型生成token的动作。  
轨迹trajectory： 轨迹由一系列的状态和动作组成，一条完整的轨迹就是一次完整的采样，相当于大模型生成一条完整的句子  
奖励Reward：做出一个动作时，得到的奖励，此处通过奖励模型计算动作的奖励值。  
回报Return: 从当前时间点到结束时获得的reward的累积和，此处通过价值模型计算回报值。  
### 训练目标
训练一个Policy神经网络π，在所有的状态S下给出相应的Action，得到的Return的期望最大 === 在所有的Trajectory中，得到的Return期望最大。

## 优化函数推导
### 梯度策略算法推导
优化目标为：该策略（参数为theta，策略为τ）下所有轨迹的回报和期望最高，即  
$E(R(\tau))_ {\tau\sim P_{\theta}(\tau)}=\sum_{\tau}^{} R(\tau) P_{\theta}(\tau) $  
由于迭代对象为参数theta，现在两边对策略的参数theta求导  
$\nabla E(R(\tau))_ {\tau\sim P_{\theta}(\tau)}=\sum_{\tau}^{} P_{\theta}(\tau)R(\tau) \frac{\nabla P_{\theta}(\tau)}{P_{\theta}(\tau)} $  
现在利用实际的N次采样，来模拟公式里的概率值，则上面的公式近似为  
$\nabla E(R(\tau))_ {\tau\sim P_{\theta}(\tau)}\approx \frac{1}{N} \sum_{n=1}^{N} R(\tau^{n}) \frac{\nabla P_{\theta}(\tau^{n})}{P_{\theta}(\tau^{n})} $  
利用log的求导公式不难得到上面的公式即  
$\frac{1}{N} \sum_{n=1}^{N} R(\tau^{n}) \nabla logP_{\theta}(\tau^{n})$  
现在再将一条完整的轨迹τ的概率，表达为一系列状态s下动作a的概率的连乘形式，其中t从开头到结尾，则上面的公式可以化为  
$\frac{1}{N} \sum_{n=1}^{N} R(\tau^{n}) \nabla log\prod_{t=1}^{T_{n} } P_{\theta}(a^{t}_ {n}\mid s^{t}_ {n} )$  
再将连乘提出log，变为连加得到  
$\frac{1}{N} \sum_{n=1}^{N} R(\tau^{n}) \sum_{t=1}^{T_{n} } \nabla log P_{\theta}(a^{t}_ {n}\mid s^{t}_ {n} ) = \frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_ {n} } R(\tau^{n}) \nabla log P_{\theta}(a^{t}_ {n}\mid s^{t}_ {n} )$  
这就是梯度策略算法Policy gradient的表达式，在神经网络迭代时，就使用这个公式来计算loss, 去掉对梯度的求导，得到整理后的表达式  
$\frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_ {n} } R(\tau^{n}) log P_{\theta}(a^{t}_ {n}\mid s^{t}_ {n} )$  
观察可知该函数由一个return和一个概率两部分相乘得到，由于对数函数的底数为e，是一个单调递增的函数，所以这个表达式的直观含义是，当Return小的时候，我们要减小轨迹的概率，而Return大的时候，我们要增大这个轨迹的概率。  

### 梯度策略算法优化
上面得到的公式存在几个问题，首先就是我们计算的是整条轨迹的return，而整条轨迹中与当前时刻动作相关的只有当前时刻之后的部分，且当前离当前时刻的动作越接近，影响应该越大，为此我们修改整条轨迹的return如下，只计算该动作时刻之后的return，其中γ小于1，用于模拟随时间的影响衰减  
$R(\tau^{n}) \to \sum_ {t'=t }^{T_ {n} } \gamma ^{t'-t} r^{n} _ {t'} = R^{n} _ {t} $  
另外一个问题是，在好的局势下，不论哪个动作都会获得正的reward，算法就会增加所有动作的概率，而坏的局势下，不论哪个动作都会获得坏的reward，算法就会减少所有动作的概率，这样会影响到收敛的速度。所以我们需要给所有动作的reward都减去一个baseline，来让相对好的动作获得正的reward而相对坏的获得负的reward。  此时我们的公式就变成了  
$\frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_ {n} } ( R^{n} _ {t}-B(s^{t}_ {n}  )) \nabla log P_{\theta}(a^{t}_ {n}\mid s^{t}_ {n} )$

### 进一步优化
1.Action-Value Function 动作价值函数  
 $R^{n} _ {t} $是通过随机采样计算得到的，方差很大，会导致训练不稳定。 使用动作价值函数替代，其意义是在State s下，做出动作Action a，期望的回报。  
$Q_{\theta}(s,a)$  
2.State-Value Function 状态价值函数  
我们上面提到的baseline，可以用该状态s下，期望得到的回报来表示，这就是状态价值函数。  
$V_{\theta}(s)$  
3.Advantage Function 优势函数  
将上面两个相减，就可以替换原来公式里对应的部分，替换后的这个部分就是优势函数，他表示状态s下做出动作a，相比其他动作可以带来多少优势。  
$A_{\theta}(s,a)=Q_{\theta}(s,a)-V_{\theta}(s) $  
替换到原来的公式中就得到了  
$\frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_ {n} } A_{\theta}( s^{t}_ {n},a^{t}_ {n}) \nabla log P_{\theta}(a^{t}_ {n}\mid s^{t}_ {n} )$  
我们对优势函数进行进一步推导，易得当前的动作价值函数，等于当前动作的reward加上下一个状态的状态价值函数乘以衰减系数，即：  
$Q_{\theta}(s_ {t},a) = r_ {t} + \gamma * V_{\theta}(s_ {t+1})$  
带入公式得到  
$A_{\theta}(s_ {t},a) = r_ {t} + \gamma * V_{\theta}(s_ {t+1})-V_{\theta}(s_ {t}) $  
我们可以对t+1时刻的状态价值函数进行采样估算，将其替换掉    
$V_{\theta}(s_ {t+1}) \approx r_ {t+1} + \gamma * V_{\theta}(s_ {t+2})$  
依次类推，我们还可以继续对t+2时刻的状态价值函数进行上面的采样估算，也可以到t+3,t+4....，由于状态价值函数本身是用一个神经网络拟合期望，他是一个有偏估计，特点是方差小偏差大，而采样的特点是偏差小方差大，所以采样的步数越多，结果的方差就会越大，而偏差会越小，反之同理。  
用下面的公式表示  
$\delta ^{V} _ {t}  = r_ {t} + \gamma * V_{\theta}(s_ {t+1})-V_{\theta}(s_ {t}) $  
则每次采样的优势函数可以表达为  
$A^{1}_ {\theta}(s_ {t},a) = \delta ^{V} _ {t} $  
$A^{2}_ {\theta}(s_ {t},a) = \delta ^{V} _ {t} +   \gamma * \delta ^{V} _ {t+1} $  
$A^{3}_ {\theta}(s_ {t},a) = \delta ^{V} _ {t} +   \gamma * \delta ^{V} _ {t+1}  +   \gamma^{2} * \delta ^{V} _ {t+2}$
