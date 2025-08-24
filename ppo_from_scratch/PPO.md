# PPO数学解析
## 基础概念
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

