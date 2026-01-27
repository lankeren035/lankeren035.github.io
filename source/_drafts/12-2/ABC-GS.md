已有的 3D 风格迁移（尤其是 NeRF-based）基本用 **NNFM 损失**：完全是 **局部 patch 对 patch**

用 3DGS 的显式高斯表示，做更容易控制的 3D 风格迁移, 让结果更忠实地对齐到**风格图的全局风格分布**；用户可以控制：

- 单一风格图；
- 多张风格图组合（compositional）；
- 单张图里不同区域 → 场景里不同语义（semantic-aware）。





 阶段一：Controllable Matching Stage

3.1 Mask Match：内容 / 风格 / 高斯的语义对齐

1. 用户对内容视角图跑 SAM 得到 mask，或者手工划分；每个 mask 对应一个语义标签（天空、树、地面…）。对于第 $i$ 个高斯和第 $l$ 个语义标签，定义一个权重 $w_i^l$：表示“这个高斯有多大概率属于语义 l”。

	1. 把高斯投到所有带内容 mask 的视角
	2. 对每个被这个高斯覆盖到且 mask_label = l 的像素，累积贡献
	3. 归一化后得到 $w_i^l$
	4. 然后设一个阈值：如果 $w_i^l > \tau$，就认为高斯 $i$ 属于语义标签 $l$。
	5. 每个高斯就有一个或多个语义标签 

2. 对风格图，同样可以画/分割 mask（语义区域：比如“天空风格区域、地面风格区域”等。内容 mask 与风格 mask 之间按语义（或用户指定）做匹配。定义第 $z$ 个语义匹配组：
	$$
	\mathcal{G}_z = \{\text{content mask}_z, \text{style mask}_z, \text{属于这个语义的高斯集合}\}
	$$
	所有后续的颜色匹配和风格损失，都在每个 $\mathcal{G}_z$ 里单独算

	- 怎么匹配：style masks 和 content masks 可以通过语义关系或用户手动进行匹配。没细说

3.2 Style Isolation：避免风格“串味”

直接用分割 mask（哪怕腐蚀一下）去裁剪风格特征，会出现 **style leaking**：

- VGG-16 有很大的感受野，某个位置的 feature 会混入邻近语义的纹理；
- 只靠像素级 mask 或腐蚀后的 mask 去“抠 feature map”不够干净。

对每个语义组 $z$ 的风格 mask $M^s_z$：

1. 在原风格图上，只保留这个 mask 区域的像素，把其它区域挖空；
2. 为了避免“黑洞”，挖空区域用：
	- 镜像、
	- 平移复制、
	- 单色填充
		 等方式补全，让整张图看起来完整，但**其它语义的原始风格被抹掉**。
3. 用这张“只保留目标风格区域”的新图丢进 VGG 提取特征：
	- 只取 mask 区域对应的特征作为该语义组的 style features $F^s_z$。

3.3 Color Match：线性色彩对齐 + 重建

在真正做特征风格迁移之前，先做一步**颜色统计匹配**（类似经典 color transfer）：

1. 取内容图中所有像素颜色集合 $X_c$，风格图所有像素颜色集合 $X_s$；

2. 求各自的均值 $\mu_c,\mu_s$ 和协方差矩阵 $\Sigma_c,\Sigma_s$；

3. 解一个线性变换：

	找到矩阵 $A$ 和偏置 $b$，使得：
	$$
	\begin{cases}
	A \mu_c + b = \mu_s \\
	A \Sigma_c A^\top = \Sigma_s
	\end{cases}
	$$
	这样，对任意内容颜色 $x$ 做
	$$
	x' = A x + b
	$$
	后，其均值/协方差就对齐到了风格图。

4. 对**内容图像像素**和**所有高斯的颜色**都做同样的线性变换：

- 内容图（GT） → 得到“颜色迁移后的内容图”；
- 高斯颜色 → 得到“颜色也被硬变换过的 3D 场景”。

5. 此时，变换完的内容图和用高斯渲染出来的图 **并不完全一致**（因为 3D 投影、可见性等原因）。于是再做一步：

- 以“颜色变换后的内容图”为监督；
- 用原 3DGS 的重建损失（L1 + SSIM 那类）去微调高斯参数，让渲染结果贴近这个新的内容图（式 (4)）。



阶段二：Stylization Stage

主要包含三类损失：

1. FAST 风格损失（核心贡献）
2. 内容保护损失
3. 几何保护（深度 + 高斯正则）



### 4.1 特征提取与语义分组

每次迭代：

1. 用当前的 3DGS 参数，从某个训练视角渲染出图像 $I_r$；
2. 用预训练 VGG-16 conv3 block 提取特征（他们使用 conv3 block 的 feature map）；
3. 对应的风格图（或隔离风格图），也用同一个 VGG 提特征；
4. 利用语义匹配组 $\mathcal{G}_z$ 的 mask，把：
	- 渲染特征图分成多个子集 $F^r_z$（只保留语义 z 区域的特征向量）；
	- 风格特征分成多个子集 $F^s_z$。

后面所有 FAST 计算，都是在每个语义组 z 上独立进行。

### 4.2 FAST：Feature Alignment Style Transfer Loss

**对比 NNFM：**

- NNFM：
	- 对每个渲染特征 $f_i$，在 style feature 中找最近邻 $g_{NN(i)}$；
	- loss = Σ d_cos($f_i, g_{NN(i)}$)；
	- 每个像素只看自己的 patch，互相之间没有“协调感”。

**FAST 的核心思路：**

> 不再对每个点独立找最近邻，而是
>  用“所有 KNN 对”的关系，先求出一个**全局对齐映射**（alignment matrix），
>  再让渲染特征整体向风格分布对齐。

具体来说，对某个语义组 z：

1. **记特征集合**

	- 渲染特征：$F^r_z = \{\mathbf{f}_i\}_{i=1}^{n_r}$
	- 风格特征：$F^s_z = \{\mathbf{g}_j\}_{j=1}^{n_s}$

2. **构造邻域关系 & Affinity Matrix $W$**（式 (5)）

	他们定义了一个邻域关系：

	- 对每个渲染特征 $\mathbf{f}_i$，在风格特征集合里找 k 个最近邻（用归一化 cosine 相似度）；
	- 对每个风格特征 $\mathbf{g}_j$，在渲染特征集合里也找 k 个最近邻；
	- 这样会得到一堆“邻居对”：$(\mathbf{f}_i,\mathbf{g}_j)$；
	- 用这些邻居对构成一个 **affinity 矩阵 $W_z$**，其元素 $W_{ij}$ 与 “$\mathbf{f}_i$ 与 $\mathbf{g}_j$ 的相似度”以及是否在彼此的 KNN 之中有关。

	简单理解：

	- NNFM：**每个点只认一个邻居**；
	- FAST：**收集所有 KNN 关系**，这些关系共同决定“渲染分布应该整体如何扭成风格分布”。

3. **求 Alignment Matrix $P_z$**（式 (6)）

	他们设一个优化目标，大致是：

	> 找一个线性变换矩阵 $P_z$，
	>  让所有邻居对 $(\mathbf{f}_i,\mathbf{g}_j)$ 在变换后尽量靠近。

	形式上类似于一个加权最小二乘：
	$$
	\mathcal{L}_{align}(P_z) 
	= \sum_{(i,j)\in\text{邻居对}} W_{ij} \, \| P_z \mathbf{f}_i - \mathbf{g}_j \|^2
	$$
	通过式 (6) 可以解析地解出 $P_z$（具体推导在附录）。直觉上，$P_z$ 把“渲染特征分布”整体拉扯到“风格特征分布”上。

	**得到对齐后的特征 $\tilde F^r_z$**

	有了 $P_z$ 后，把所有渲染特征做变换：
	$$
	\tilde{\mathbf{f}}_i = P_z \mathbf{f}_i
	$$
	得到一个**对齐后的特征图** $\tilde F^r_z$，它在分布上应该更接近风格特征集合 $F^s_z$。

	**FAST 风格损失（式 (7)）**

	他们的 FAST loss 的描述是：

	> “通过最小化渲染特征与对齐特征之间的 cosine 距离来实现风格迁移。”（见 Fig.3 caption）

	换句话说，对每个语义组 z：
	$$
	\mathcal{L}^{(z)}_{FAST} 
	= \frac{1}{n_r} \sum_{i=1}^{n_r}
	  d_{\cos}(\mathbf{f}_i, \tilde{\mathbf{f}}_i)
	$$
	全局 loss 就是所有组的和：
	$$
	\mathcal{L}_{FAST} = \sum_z \lambda_z \mathcal{L}^{(z)}_{FAST}
	$$
	（论文具体形式写在式 (7)，但精髓就是让“当前渲染特征”往“对齐后的目标特征”靠。）

**和 NNFM 的本质区别：**

- NNFM：
	- 每个像素独立找最近邻 → 只用到一小部分 style 特征；
	- 不考虑不同像素之间的关系；
- FAST：
	- 利用所有 KNN 关系，先求一个全局对齐映射；
	- 同时考虑**整个特征分布**和像素间的关系；
	- 更容易覆盖风格图的**全局风格**（颜色、纹理占比等）。

### 4.3 内容与几何保护损失

为了不把场景“画残”，他们加了几类约束（IV-C 后半段）。

#### 4.3.1 内容保持损失（Content Loss, 式 (8)）

- 用同一套 VGG 提取渲染结果的特征 $F^r$ 和原内容图特征 $F^{cont}$；

- 内容损失：
	$$
	\mathcal{L}_{content} = \frac{1}{N} \sum_{i} \|F^r_i - F^{cont}_i\|_2^2
	$$

- 防止风格化之后完全看不出原场景内容。

他们还加了一个 **Total Variation (TV) 损失**，减少高频噪声、边缘毛刺。

#### 4.3.2 深度损失（Depth Loss, 式 (9)(10)）

- 利用 3DGS 的几何明确性：

	- 对原始高斯（未风格化）渲染一个**初始深度图** $D_0(x)$；
	- 对当前（风格化过程中的）高斯渲染深度图 $D(x)$。

- 深度损失：
	$$
	\mathcal{L}_{depth} 
	= \frac{1}{N} \sum_x |D(x) - D_0(x)|
	$$

- 这会强烈约束几何别乱动：高斯的位置/尺度/不透明度的改变不能破坏原有深度结构。

#### 4.3.3 高斯正则（Scale & Opacity Regularization, 式 (11)）

- 在训练时，如果不约束，3DGS 容易出现：

	- 高斯无限变大变细，形成“针状高斯”；
	- 或者过多重叠导致图像模糊。

- 因此，他们用原始高斯参数 $(s_i^0,\alpha_i^0)$ 做正则：
	$$
	\mathcal{L}_{reg} = \sum_i \big( \|s_i - s_i^0\|^2 + \|\alpha_i - \alpha_i^0\|^2 \big)
	$$

- 再配合关闭 densification（不再新增高斯，文中说明在 stylization 阶段禁用 densification），整体就能比较好地**锁住几何结构，只改外观**。

### 4.4 总损失 & 训练什么参数？

在 stylization 阶段，总损失大致是：
$$
\mathcal{L}_{total}
= \lambda_{FAST} \mathcal{L}_{FAST}
+ \lambda_{content} \mathcal{L}_{content}
+ \lambda_{TV} \mathcal{L}_{TV}
+ \lambda_{depth} \mathcal{L}_{depth}
+ \lambda_{reg} \mathcal{L}_{reg}
$$
他们提到：

- 使用 VGG-16 conv3 block 作为特征提取；
- FAST 里的 k=5；
- Stylization 阶段**禁用 densification**。

论文没有逐项罗列“只训练哪些参数”

## 5. 可控性：三种风格迁移模式

得益于 **Mask Match + Style Isolation**，ABC-GS 能支持三种风格设置（Fig.1, Fig.5）：

1. **Single-Image Style Transfer**
	- 一张风格图；
	- style mask 覆盖整图；
	- 所有语义组共享同一个风格统计 → 整个场景统一风格。
2. **Compositional Style Transfer**
	- 多张风格图；
	- 不同内容区域（天空、建筑、路面…）对应不同风格图 + mask；
	- 通过 semantic matching group，控制“哪里用哪张风格”。
3. **Semantic-Aware Style Transfer**
	- 一张风格图里有多个语义区域；
	- 对风格图提多个 mask（+ style isolation），每个 mask 对应一种局部风格；
	- 把这些 mask 对应到内容场景的不同语义组。

本质上，**所有可控性都由“你如何定义和匹配语义组 G_z” 决定**。





