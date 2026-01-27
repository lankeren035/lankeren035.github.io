用于高分辨率风格迁移？

我们证明了一种基于全局神经统计的新多尺度损失（我们将其命名为 SOS，即同时优化尺度），可以将风格迁移到超高分辨率 3D 场景。SGSST 不仅在如此高的图像分辨率下开创了 3D 场景风格转换的先河，而且通过彻底的定性、定量和感知比较评估，它还产生了卓越的视觉质量。

这两个框架都激发了对 3D 风格传输算法的尝试 [6,16,22,25,26,33,46,48]，但这些方法到目前为止只产生了中等分辨率的输出。它们不能忠实地传输高分辨率多尺度纹理，例如绘画中存在的纹理。

- 我们引入了SOS，一种以单一无参数且可解释的公式表示的同步优化尺度损失。
- 通过单独优化SOS 损失，我们达到了3DGS 样式传输的UHR，并且我们将高斯泼溅样式传输缩放了四倍的分辨率增益。
- 卓越的质量转移：通过转移大量的全球风格统计数据，我们即使在人力资源解析中也能获得卓越的风格转移质量

我们的方法是第一个允许 UHR 风格直接转移到 3DGS 的方法。它产生高视觉质量的结果，并依赖于优化单个多尺度损失。

基于优化，SGSST 的主要限制是训练时间相当长，比初始 3DGS 训练长两到八倍，具体取决于图像分辨率。

由于 AdaIn [26, 33]，风格化速度很快但非常近似；或者由于 NNFM [22, 46, 48]，无法使用 HR 风格图像。据我们所知，SGSST 是第一个允许在 UHR 训练和渲染的 3DGS 进行高质量传输的程序。



###  传统方法

在 2D 风格迁移里，常见做法是：[arXiv](https://arxiv.org/html/2412.03371)

1. 先在低分辨率图上优化到收敛；
2. 把结果上采样到更高分辨率，接着优化；
3. 一直加分辨率，直到目标分辨率。

但是作者发现，在 3DGS 的场景中，这种 coarse-to-fine 策略会出现：

- 在低分辨率学到的“大块色调、大笔触”
	 → 在高分辨率阶段会被小尺度纹理的优化**慢慢抹掉**；
- 最终结果只剩下局部纹理，没有明显的大尺度风格结构。[arXiv](https://arxiv.org/html/2412.03371)

他们把这个现象在实验 Section 5.3 里展示出来：随着逐层提升分辨率，大尺度损失反而变大，对应大笔触消失。[arXiv](https://arxiv.org/html/2412.03371)

原因之一是：
 3DGS 是一组稀疏高斯，不像像素网格那样弹性大，一个尺度上的改动会对其他尺度产生很强的耦合，coarse-to-fine 难以“锁住”已学到的结构。[arXiv](https://arxiv.org/html/2412.03371)







### 通常损失：

x -> F -> m -> l


$$
\frac{\partial l}{\partial x} = \frac{\partial l}{\partial m} \frac{\partial m}{\partial F} \frac{\partial F}{\partial x}
$$

- 要存这三个梯度



### 本文操作：

$$
x  [3,H,W] \\\\
F [C,h,w] -> [C,N]  \\\\
m.G  -> [C,C] \\\\
m.mean  -> [C] \\\\
m.sigma  -> [C]
$$

**Gram 矩阵**：
$$
G(x)
= \frac{1}{N_L} \tilde F(x)\,\tilde F(x)^\top
\in \mathbb{R}^{C_L \times C_L}
$$
**通道均值**（对空间平均）：
$$
\mu_c(x)
= \frac{1}{N_L}\sum_{p=1}^{N_L} \tilde F_{c,p}(x)
\quad\Rightarrow\quad
\mu(x)\in\mathbb{R}^{C_L}
$$
**通道标准差**：
$$
\sigma_c(x)
= \sqrt{\frac{1}{N_L}\sum_{p=1}^{N_L}
  \big(\tilde F_{c,p}(x)-\mu_c(x)\big)^2}
\quad\Rightarrow\quad
\sigma(x)\in\mathbb{R}^{C_L}
$$
 **这一层的风格损失**

用 Gram + mean + std 的组合（忽略前面的权重符号）：
$$
\begin{aligned}
E^{(L)}_{\text{style}}(x;v)
=&\ \alpha_G^{(L)}
    \big\|G(x) - G(v)\big\|_F^2 \\
&+ \alpha_\mu^{(L)}
    \big\|\mu(x) - \mu(v)\big\|_2^2 \\
&+ \alpha_\sigma^{(L)}
    \big\|\sigma(x) - \sigma(v)\big\|_2^2
\end{aligned}
$$

## 针对某一层 L：从 loss 到特征的梯度（形状 + 公式）

现在我们只看某一层 $L$ 的梯度  $\dfrac{\partial E^{(L)}_{\text{style}}}{\partial \tilde F^{(L)}(x)}$，

### Gram 项的梯度

Gram 部分：
$$
E_G = \alpha_G \,\lVert G - G^\* \rVert_F^2
$$

1. **对 Gram 本身的梯度**：

$$
\frac{\partial E_G}{\partial G}
= 2\alpha_G\,\Delta G
$$

**Gram 对特征的导数**：
 $G = \frac1{N_L}\tilde F\tilde F^\top$，
$$
G_{ij} = \frac1{N_L}\sum_{p=1}^{N_L} \tilde F_{i,p} \tilde F_{j,p}
$$
对单个元素 $\tilde F_{c,q}$ 的偏导：
$$
\frac{\partial G_{ij}}{\partial \tilde F_{c,q}}
= \frac{1}{N_L}
 (\delta_{i,c}\tilde F_{j,q} + \delta_{j,c}\tilde F_{i,q})
$$