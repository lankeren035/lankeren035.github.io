---
title: 'SODA: An Adaptive Bitrate Controller for Consistent High-Quality Video Streaming'

date: 2025-4-19

tags: [论文]

categories: [论文]

comment: true

toc: true


---

#

<!--more-->



# 0. Abstract

- 自适应码率（ABR）流的首要目标是通过根据网络状况动态调整视频码率来提升用户体验（QoE）。然而，在网络波动时（尤其是直播场景下缓冲区较短），频繁的码率切换会导致用户对视觉质量的不一致感到困扰。本文提出了一种既可部署又能显著平滑视觉质量波动的“平滑优化动态自适应”（SODA）控制器。SODA不仅具备严格的理论性能保证，在数值仿真中相较于最优基线提升了 9.6%–27.8% 的 QoE，在原型环境下提高了 30.4%，在 Amazon Prime Video 的生产环境中减少了多达 88.8% 的码率切换，并将用户平均观影时长提升了 5.9%。为保证可大规模部署，SODA 在多项式时间内完成“码率前瞻规划”，相较于指数复杂度的暴力搜索，实测仅需约 200 次迭代即可出结果。 

# 1. 引言（Introduction）

- 随着在线视频的普及，用户使用的观看设备非常多样：笔记本、手机、智能电视、机顶盒、游戏主机等，这些设备的硬件能力和网络接入方式（Wi‑Fi、蜂窝、宽带等）均不同。为保证所有用户的高质量体验（QoE），视频服务商采用自适应码率（ABR）技术：将同一视频以不同码率（如 720p、1080p、1440p……）分别编码，并将视频切分成若干短片段（例如每段 2 秒）；客户端播放器根据实时网络和设备情况为每个片段选择合适码率，并将下载到的片段存入缓冲区后再播放。

- 已有研究表明，用户 QoE 取决于三大要素：

1. **视频质量**（码率越高、分辨率越大，画面更清晰）；
2. **重缓冲时长**（播放中断越少）；
3. **码率切换频率**（切换越频繁，视觉体验越不稳定）。

例如，重缓冲时间每增加 1%，平均观影时长就会减少近 3 分钟，而频繁的码率切换会显著提高用户放弃观看的概率¹。

相比点播（on‑demand）场景，**直播** 缓冲区一般只保留 10–20 秒内容（以紧跟直播源），而点播可达 60–180 秒。这使直播对重缓冲和码率切换更敏感——较短的缓冲区意味着即便微小的带宽波动也会触发频繁切换。图 1 显示，在一个大型体育直播事件中，若码率切换率超过 20%，用户仅会观看不足 10% 的总时长。

本论文针对这一难题，提出了 SODA 控制器，主要贡献包括：

1. **理论基础**：首次将“画质”、“重缓冲”与“切换平滑”三者纳入同一优化框架，并证明算法在在线情况下（无未来带宽完美预知）仍能接近离线最优。
2. **仿真与原型评估**：数值仿真 QoE 提升 9.6%–27.8%；原型评估 QoE 提升 30.4%。
3. **生产部署**：在 Amazon Prime Video 真实用户上，码率切换减少 88.8%，平均观看时长提升 5.91%。
4. 对吞吐量预测错误的鲁棒性。
5. **多项式时复杂度**：设计近似求解器，在可接受的 200 次迭代内完成前瞻规划，适合资源受限设备。



# 2

# 3. 方法

## 3.1 公式1

- $\omega_ n$： **平均网络吞吐量**（bandwidth），表示在该时段内，客户端实际可用的下载速率，单位通常为 Mb/s 
- $r_n$： **所选码率**（bitrate），播放器为该时段选择的编码码率，同样以 Mb/s 为单位。 
- $x_n$： **缓冲区时长**（buffer level），表示到时刻 n 时，播放器内已下载但尚未播放的视频时长（秒）。 
- $r_ {min}, r_ {max}, x_ {max}$分别为：可选码率的上下限，缓冲区最大容量（秒）

$$
\sum_{ n = 1 } ^ N \big( v ( r_ n) \cdot \frac{\omega_n \Delta t}{r_ n} + \beta \cdot b( x_ n ) +\gamma \cdot c( r_n , r_ { n-1 }) \big)
$$

- ### 1) 画质代价：$v ( r_ n) \cdot \frac{\omega_n \Delta t}{r_ n}$

  - $v(r)$： **失真成本函数**（distortion cost），要求是“正的、随着 r 增大严格下降且凸”的函数，用于模拟编码扭曲, 例如$ 1/r$ 或 $log(r_{max} / r)$。 

  - $\frac{\omega_n \Delta t}{r_ n}$ ： 在时段 n 内实际下载的视频时长（秒）。分子为下载量，除以每秒 Mb 即码率 $r_ n$ 可得下载的视频时长。 

  -  **含义**：$v$越小（码率越高，画质越好），但同时下载量固定时，高码率要更多带宽开销。两者乘积即“该时段内因选择$r_ n$而产生的总失真代价”。 

    > **直观**：越高码率、失真越低；但若带宽不足（$\omega_ n < r_n$），下载时长会受限。该项鼓励使用高码率。 
    >
    >  因为每个时间格下载的视频量是 ω^ Δt\hat\omega\,\Delta tω^Δt，而每单位视频需要  ⁣r ⁣/(ω^ Δt)\!r\!/( \hat\omega\,\Delta t)r/(ω^Δt) 的码率，所以真实画质损失要乘以 ω^ Δt/r\hat\omega\,\Delta t/rω^Δt/r。 

- ### 2)  缓冲稳定代价：$\beta \cdot b(x_ n )$

  - $\beta > 0$ : 用户偏好权重，越大越重视保持稳定缓冲。 

  - $b(x)$： **缓冲成本函数**，围绕一个“安全目标缓冲水平 $\bar x$ 做二次惩罚： 

    $$b(x_ n ) = \left \{ \begin{matrix} ( \bar x - x_ n )^ 2 &x_ n \leq \bar x  \\\\ \epsilon (x_ n - \bar x )^ 2 &x_ n > \bar x , 0< \epsilon < 1  \\\end{matrix}\right.$$
    
    -  当$x_ n \le \bar x$ 时，缓冲不足，惩罚较重； 
    -  当 $x_ n > \bar x$时，缓冲过多，惩罚较轻（因超额缓冲虽能防止重缓冲，但会增加延迟和带宽占用，故轻罚）。 
    
    > **目的**：间接“最小化重缓冲”。
    > 直接对“零缓冲”做硬惩罚会导致非凸、难以优化；而平滑二次惩罚既保证凸性，也让缓冲接近非零安全区。 

- ### 3) 切换代价：$\gamma \cdot c( r_n , r_ { n-1 })$

  - $\gamma > 0$： 用户偏好权重，越大越不喜欢频繁切换。 

  - $c( r_n , r_ { n-1 })$ :  **码率切换成本**，例如可以取 :$( v(r_ n) - v( r_ { n - 1 }) )^ 2$ 或 $( r_ n -  r_ { n - 1 } )^ 2$  或其他单调递增度量。 

    > **含义**：若相邻时段码率变化大，带来视觉抖动，代价增大 
    >
    > **作用**：在保证画质和缓冲的同时，**平滑**观感——避免“画面突然清晰↔模糊”带来的不适。 

## 3.1 公式2

-  为了让$x_ n$ 正确演化，需满足： 

$$
x_ n = x_ { n - 1 } + \frac{ \omega_ n \Delta t}{ r_ n } - \Delta t , x_ n \in [ 0 , x_ {max } ]
$$



- ### 1）缓冲动态约束
  -  **下载项** $\frac{ \omega_ n \Delta t}{ r_ n }$该时段新增缓冲。 

  -  **播放项** $\Delta t$：用户在该时段持续播放 Δt 秒视频，消耗同等缓冲。 

    > **边界**：如果$x_ n$掉到 0，就发生重缓冲（空缓冲）；如果超过 $x_ {max}$，则浪费或增加延迟。 



- 为什么采用基于时间的公式？
- 为何不直接建模重缓冲？

## 3.2 纳入吞吐量预测

- 上面的公式需要知道每个时刻的吞吐量（带宽），正确预测每个时间的吞吐量至关重要。

- SODA 的设计与传统的**段为基础**的自适应码率（ABR）控制不同。传统的 ABR 控制器（如 MPC 和 Fugu）常常将带宽预测与码率选择交织在一起，这导致了 **因果依赖问题**。具体而言，它们的带宽预测有效期依赖于所选择的码率。即当选择低码率时，带宽预测有效的时长较短；而选择高码率时，预测的有效期则较长。这样一来，带宽预测就受到决策的影响，难以保持一致性，尤其是在带宽波动较大的环境下。
- SODA 通过 **时间为基础的建模** 设计来解决这一问题，它将每个时间区间的带宽预测独立于所选码率进行处理。这意味着，SODA 选择的码率不会影响带宽预测的有效期，进而避免了因果依赖的问题。

## 例子设置

- **段时长** L=2L=2L=2 秒
- **SODA** 预测窗口：未来 K=3K=3K=3 个时间格，每格长度 Δt=2\Delta t=2Δt=2 秒，总共 **6 秒**
- **MPC** 预测窗口：未来 K=3K=3K=3 段
- 假设带宽预测恒定：ω=3\omega=3ω=3 Mb/s
- 当前缓冲后上一次选的码率（即 rn−1r_{n-1}rn−1 或 MPC 上一次的初始码率） = **2 Mb/s**

我们让两者都做 “下一步码率” 的决策。

------

### 一、SODA（时间为基础）

1. **预测输入**
    SODA 拿到一个长度为 KΔt=6K\Delta t=6KΔt=6 秒的带宽预测：

   3,3,3⏟每个 2 秒 格  Mb/s  \underbrace{3,3,3}_{\text{每个 2 秒 格}} \;\text{Mb/s}每个 2 秒 格3,3,3Mb/s

2. **固定窗口**
    无论之后它选什么码率，这 6 秒内的预测都“有效”——即它始终在一个**恒定的时长窗口**上优化。

3. **一次优化、一阶决策**
    SODA 在这 6 秒里用式 (2a–c) 的代价函数：

   ∑m=nn+2[v(rm)3⋅2rm+β b(xm)+γ c(rm,rm−1)]  \sum_{m=n}^{n+2}\Bigl[v(r_m)\frac{3\cdot2}{r_m}+\beta\,b(x_m)+\gamma\,c(r_m,r_{m-1})\Bigr]m=n∑n+2[v(rm)rm3⋅2+βb(xm)+γc(rm,rm−1)]

   求出 (rn,rn+1,rn+2)(r_n,r_{n+1},r_{n+2})(rn,rn+1,rn+2) 最优，然后**只执行 rnr_nrn**，并在下一时刻再更新带宽预测（仍然是未来 6 秒）。

> **关键**：不管 SODA 选低码率还是高码率，它**都用同样的 6 秒预测**来做决策，预测窗口长度不受 rmr_mrm 的影响。

------

### 二、MPC（段为基础）

1. **预测输入**
    MPC 也拿到未来 3 段的带宽预测，每段预测为 3 Mb/s。但 **“这些预测能持续多久”**，取决于你下载这 3 段共花了多少时间。
2. **下载时间依赖码率**
   - 如果 MPC 在这 3 段都选 **低码率** ri=1r_i=1ri=1 Mb/s：
      每段大小 =1 ⁣× ⁣2=2=1\!\times\!2=2=1×2=2 Mb，下载时长 2/3≈0.672/3≈0.672/3≈0.67 s，3 段合计 ≈2.0 s
   - 如果 MPC 都选 **高码率** ri=3r_i=3ri=3 Mb/s：
      每段大小 =3 ⁣× ⁣2=6=3\!\times\!2=6=3×2=6 Mb，下载时长 6/3=26/3=26/3=2 s，3 段合计 =6.0 s
3. **预测窗口随决策改变**
    因此，MPC 假设“下 3 段的带宽预测有效期”要么 **2 秒**（若选低码率），要么 **6 秒**（若选高码率），中间还可能出现其它组合。
4. **一次优化、一阶决策**
    MPC 也是在未来 3 段上优化同样的 QoE 代价，但它的**时域长度**却跟它要选的 rir_iri​ 互相绑定。

> **问题来了**：MPC 的预测模型得假设“这 3 段下载期间带宽不变”，却又不知道“下载这 3 段到底是 2 秒、4 秒还是 6 秒”？预测有效性的前提就被破坏了。



## 3.3. 控制机制

-  **如何将前面基于“时间格”的建模，转化为一个可执行的 “模型预测控制”（MPC）式的在线算法**。 

### 1）优化目标

$$
\sum_{ m = n } ^ {n + K - 1} \big( v ( r_ m) \cdot \frac{ \hat \omega_{ m | n - 1 } \Delta t}{r_ m} + \beta \cdot b( x_ m ) +\gamma \cdot c( r_ m , r_ { m-1 }) \big)
$$

### 2）动态缓冲约束

$$
\left \{ \begin{matrix} x_ m = x_ { m - 1 } + \frac{ \omega_ { m | n - 1 } \Delta t}{ r_ m } - \Delta t \\\\ x_ m \in [ 0, x_{ max }, r_ m \in R ] \end{matrix} \right.
$$

### 3）滚动执行 ：只取首个决策 

-  这个优化是在“当前时刻n **前瞻** K 格 ，求出一条最优码率序列 （$r_n , r_{n+1}, \cdots,r_{n+K-1}$），然后只执行$r_ n$，进入下一时刻n+1再重新计算

# 4. 理论设计洞察

### 4.0.1 背景与目标

- SODA 的设计借鉴了两个方向的最新进展：
  1.  平滑在线凸优化（SOCO） 
     -  在 SOCO 中，每次决策不仅要最小化即时损失（如画质损失），还要对连续决策之间的“切换”进行惩罚，以避免频繁震荡。 
     -  SODA 将视频的三大 QoE 组件（画质、重缓冲、码率切换）对应到 SOCO 的“即时损失 + 切换惩罚”框架中。 
  2.  在线控制／模型预测控制（MPC） 
     -  MPC 会在每一步使用未来若干步的预测来求解一个小规模优化，并只执行第一步决策；下一步再重做优化。 
     -  SODA 采用类似策略，但在理论分析中引入了 **指数衰减扰动** 的概念来量化预测误差的影响。 

### 4.0.2 指数衰减扰动属性

- 定义（非形式化）

> **指数衰减扰动** 意味着：当**固定未来带宽预测**$\{ \hat \omega _ m \}$时，不同初始条件 $x_ {n-1}, u_{n-1}$ 下求得的最优轨迹会随着时间逐步“汇聚”(指数级收敛， 即扰动影响快速衰减 ），它们在第 n+k步的差异上界是 $\rho ^ k$量级，其中 $\rho < 1$ 是衰减因子。
> 同理，当**固定起点**时，对第 n+κ步带宽预测的微小扰动对第 n 步的最优决策影响也会呈指数衰减。
>
> - 对于u的解释：缓冲更新公式中 “非线性”(1/rₙ)会给后续的理论分析带来很大麻烦。  所以在第 4 章里，作者 **“定义动作为码率的倒数” ** ：$x_n = x _ { n - 1 } + \omega_ n \Delta t \mu _ n - \Delta t$  大大简化了后续用凸优化和在线控制理论证明性能界的难度。 

-  同理，当初始状态一样但预测值有偏差时，对当前决策 uₙ 的影响也会随着预测时刻与当前时刻的距离而**指数衰减**（见文中图 6） 

### 4.0.3 性能度量

我们用两种常见在线优化的度量来衡量 SODA：

1. **动态遗憾（Dynamic Regret）**
   $$
   Regret = cost(ALG) - cost(OPT)
   $$
   

   其中 OPT是**离线最优**（知道所有未来带宽后一次性求最优）。

2. **竞争比（Competitive Ratio）**
   $$
   cost(ALG) / cost(OPT)
   $$
   

   理想情况下 接近 1。

### 4.0.4 证明思路概览

为了给出 SODA 的遗憾与竞争比界，我们做了如下两步：

1. **界定每步误差（Per-step Error）**
   - 令 Δn表示在第 n 步，SODA 在真实状态$(x_{n-1} , u_{n-1})$下，因仅有有限预测或预测误差，与“假如知道所有未来带宽”下的离线最优状态之间的差距。
   - 指数衰减扰动属性保证，对于固定预测，Δn只依赖于前面 K步的误差，并且影响会迅速衰减。
2. **累积误差不爆炸**
   - 虽然每步 Δn会带来一点额外损失，但由于之前误差的影响会指数衰减，所有 Δn 的累加不会线性增大，而是保持在 O(1)量级。
   - 因此，SODA 整体的动态遗憾为$O(\rho ^K N)$，竞争比为$1+O(\rho ^K )$。

## 4.1 精确预测

### 4.1.1  背景：为什么要研究“精确预测”情形？ 

-  SODA 在每个时刻 n 会使用未来 K 步的带宽预测做模型预测控制（MPC）。第 4.1 节聚焦于一个理想化场景：**假设这 K 步预测完全准确（$\hat \omega_ { m | n - 1 } = \omega _ m$)，那么只要 K 是常数，SODA 的在线性能就能以指数级速度逼近离线最优。** 

### 4.1.2 **Theorem 4.1（非正式）**

当未来 K 步带宽预测 **完全准确**（$\hat \omega _ {m | n - 1} = \omega_ w$）且 K 是常数级别（K=O(1) ）时，SODA 的动态遗憾为$O( \rho ^ K N)$

> 且竞争比为$1+O(\rho ^ K)$
>
> 其中 $\rho < 1$是指数衰减因子。
>
> - **含义**：只要有准确的短期预测，SODA 就能以**指数级**逼近离线最优；K 不用很大，就可以几乎跑到最优水平。

### 4.1.3  正式定理：Theorem A.3 (附录)

在附录 A.3 中，作者给出了 Theorem 4.1 的**正式版本** Theorem A.3，明确了各种常数的取值和 K 的下界：

> **Theorem A.3.** 在 Assumption A.1（可“到达”任意缓冲水平）下，令终端约束
>
> $x_{t+K-1}=x\quad,\quad r_{t+K-1}=\hat\omega_{t+K-1\mid t-1}.$
>
> 记指数衰减因子和常数为 Theorem A.1、Corollary A.2 所给出的$ \rho,\,C,\,C'$。若所有预测精确且
>
> $K\ge\frac{1}{4}\frac{\ln\bigl(\tfrac{16}{1-\rho}(1+\tfrac{(C+C')^2}{1-\rho})(C^2+(C')^2)\bigr)}{\ln(1/\rho)}=O(1),$
>
> 则 SODA 的**动态遗憾**满足
> $\displaystyle\mathrm{regret}\le C_1\,\rho^{K-1}\,\mathrm{cost(OPT)}=O(\rho^K\,N)$
>  **竞争比**满足
> $\displaystyle\mathrm{CR}\le1+C_1\,\rho^{K-1}=1+O(\rho^K)$
>
> 其中
>
> $C_1 \;=\; 8\Bigl(2(4\gamma+\beta+\omega_{\max})\tfrac1{1-\rho}\bigl(1+\tfrac{(C+C')^2}{1-\rho}\bigr) (C^2+(C')^2)\,\tfrac{4+\omega_{\min}^2}{\epsilon\,\beta\,\omega_{\min}^2}\Bigr)^{\tfrac12}.  $

### 4.1.4 证明思路概览

要证明上述结论，关键在于将控制器在线决策偏离离线最优决策时产生的“误差”分两步控制：

#### 1 每步误差 (per-step error) 的定义与界

- **定义 A.2** 给出：在时刻 t，SODA 在有限窗口内做决策得到；$(x_t,u_t)$，而“全知离线最优”在相同初始 $(x_{t-1},u_{t-1}) $下能达到的下一个状态/动作为
  $\bigl(x_t^*,u_t^*\bigr)=\psi^N_t\bigl((x_{t-1},u_{t-1});\omega_{t:N};0\bigr)$

- **每步误差**

  $  e_t \;=\;\|x_t - x_t^*\| \;+\; \|u_t - u_t^*\|$

- **Lemma A.4**（预测准确情形）给出：

  $  e_t^2\;\le\;16\,\rho^{4K-2}(C^2+(C')^2)^2\bigl(\|x_{t-1}-x_{t-1}^*\|^2+\|u_{t-1}-u_{t-1}^*\|^2\bigr)  +8\,\rho^{2K-2}(C^2+(C')^2)\,\frac{2+\omega_{\min}^2}{\epsilon\,\beta\,\omega_{\min}^2}\,b(x_{t+K-1}^*)\,.$

  该不等式表明，每步误差会随着窗口长度 K 以 $\rho^{2K} $或更快的速度指数衰减 。

#### 2 误差累积的控制

- 即使每步都有误差，只要每步误差的影响能快速衰减，总体偏离也不会放大。

- **Lemma A.5** 证明了：

  $  \sum_{t=1}^N\bigl(\|x_t - x_t^*\|^2 + \|u_t - u_t^*\|^2\bigr)  \;\le\;\frac{1}{1-\rho}\Bigl(1+\frac{(C+C')^2}{1-\rho}\Bigr)\sum_{t=1}^N e_t^2.$

  左侧是 SODA 轨迹与离线最优轨迹在所有时刻偏差的平方和，上式保证了“过往每步误差”不会线性累积，而是被 $\tfrac1{1-\rho}$ 相关项加权 。

#### 3 合并两类误差，得出定理结论

将 Lemma A.4（个步误差指数衰减）与 Lemma A.5（误差不累积放大） 联立，可得

$  \sum_{t=1}^N\bigl(\|x_t - x_t^*\|^2+\|u_t-u_t^*\|^2\bigr)\;=\;O(\rho^{2K})\;\times\;\text{Cost(OPT)}.$

再利用系统目标函数的良态性（well-conditioned），即可把上述轨迹偏离界转化为**动态遗憾**和**竞争比**界，从而完成 Theorem A.3 的证明。

#### 5. 小结

- **Section 4.1** 的核心在于：在理想化的“精确预测”场景下，SODA 仅需常数长度的预测窗口 K，就能以**指数速率**逼近离线最优（动态遗憾 $O(\rho^K N)$，竞争比 $1+O(\rho^K)$
- 这一结论依赖于**指数衰减扰动**性质，通过对“每步误差”与“误差累积”分别建界，再合并给出总体性能界。
- 实际上，预测窗口 KKK 只要稍大于一个常数（与系统参数$ ⁡\epsilon,\beta,\gamma,r_{\min},r_{\max},x_{\max}$ 等有关），就能在精确预测情形下近乎等同于最优控制。

这样，不仅从定性角度理解了 SODA 的优势，也从定量层面把握了“KKK 为何可以是常数”的理论原因。

## 4.2 Inexact Predictions（不精确预测）

- 是SODA理论设计的关键部分之一，它探讨了SODA在带宽预测不完全准确的情况下如何保持鲁棒性和良好的性能。该节主要通过理论证明SODA如何在存在预测误差的情况下，仍能保证系统的表现不会因为这些误差而急剧下降。

### 1)  **背景：带宽预测不准确的挑战**

在实际应用中，带宽预测通常不可能做到完全精确，尤其是在动态变化的网络环境下。SODA是一个面向视频流的自适应比特率控制器，它依赖于对未来网络带宽的预测来做出决策。但实际上，由于网络波动、测量误差和预测算法的不完美，预测常常是不准确的。

这一节的目标是：**即使带宽预测存在误差，SODA依然能够维持高效的流媒体体验，减少重缓冲和比特率切换的频率**。

------

### 2) **预测误差对SODA的影响**

SODA的核心思想是平衡视频质量、重缓冲时间和比特率切换频率，并通过优化这些目标来最大化用户的QoE（体验质量）。在理想情况下，SODA会使用完美的带宽预测来做出比特率决策。然而，在实际情况中，由于带宽预测存在误差，SODA的控制决策可能会受到影响。

#### 2.1 **SODA的鲁棒性设计**

- **鲁棒性**：SODA的设计具有抗预测误差的能力。具体来说，SODA可以在带宽预测错误的情况下，依然能够较好地控制缓冲区的大小，避免出现过多的重缓冲现象，并且尽可能减少比特率的频繁切换。
- **指数衰减扰动**：这一设计的关键在于SODA利用了一种叫做“**指数衰减扰动**”（Exponentially Decaying Perturbation）的性质。这意味着，当带宽预测出现误差时，SODA对误差的敏感度会随着时间的推移而迅速减弱。也就是说，SODA的优化过程不会因短期的带宽预测误差而导致长时间的不稳定，误差的影响会在短期内迅速衰减。

#### 2.2 **误差的指数衰减性质**

- **预测误差的累积**：如果预测误差是有限的，那么SODA系统的缓冲区状态和比特率决策的偏差不会在整个时间跨度内线性增加，而是随着时间推移，预测误差对系统状态的影响将呈现指数衰减的特性。
- **局部扰动的减弱**：即使在某个时刻预测出现较大误差，SODA也能够保持对未来时刻的决策相对稳定。这个性质保证了SODA在面对短期内的不确定性时依然能够执行有效的控制。

------

### 3) **Theorem 4.2：不精确预测的鲁棒性**

在这一节中，作者给出了**Theorem 4.2**，这是该节的核心理论内容，描述了SODA在不精确预测情境下的性能保障。

> **Theorem 4.2. [Informal]**
>  假设每个预测的误差都有上界（即带宽预测误差有一定限制），SODA的缓冲区状态将始终保持在合法范围内（即不会溢出或为空），并且SODA的动态遗憾满足
>
> $\mathrm{regret} = O(\sqrt{E N} + E)$
>
> 其中 E 表示预测误差的总和，$E = \rho^{2K} N + \sum_{k=1}^{K} \rho^k E_k$其中$ E_k$是预测误差在第k步的误差量。

#### 3.1 **误差总和的定义**

- **E**：是一个总误差项，包含了所有带宽预测误差的累积，特别是在预测多个时刻时，误差会随着预测步长的增大而增大，但**指数衰减扰动**确保了这种增大的影响是有限的。
- **预测误差的增长**：随着预测步数增加，误差的增长速度趋于平缓，这就是**指数衰减**的体现。通过控制误差的扩展，SODA能够在误差积累较多的情况下仍然维持良好的QoE。

#### 3.2 **缓冲区稳定性**

- **缓冲区约束**：定理证明，SODA能够确保缓冲区始终保持在合法范围内（即 $0 < x_n < x_{\text{max}}$），这意味着不会出现严重的重缓冲（即缓冲区空了）或者过满（即缓冲区溢出）。
- **预测误差和缓冲区稳定性**：定理还表明，SODA对带宽预测误差的鲁棒性使得即使误差较大，缓冲区的稳定性也能够得到保证。这是因为SODA的设计始终以缓冲区稳定性为核心目标，即通过平稳的比特率调整来避免大幅度的波动。

------

### 4) **SODA的设计如何应对不准确的带宽预测**

SODA通过以下方式应对不准确的带宽预测：

- **鲁棒的优化目标**：在进行比特率选择时，SODA并不完全依赖于精确的带宽预测。它通过优化目标函数（包含缓冲区稳定性、视频质量和切换成本）来保证即使预测误差存在，优化过程依然能够平衡各个QoE目标。
- **缓冲区稳定性**：SODA的目标之一是稳定缓冲区的大小，而不是单纯追求准确的带宽预测。即使带宽预测存在误差，SODA仍然能够通过调节比特率，使得缓冲区保持在合理的范围内，从而避免重缓冲和比特率频繁切换。

------

### 5) **小结**

第4.2节的核心内容是：**SODA在面对不精确的带宽预测时，能够保持良好的鲁棒性和高效的流媒体表现**。这一部分通过数学证明了，尽管带宽预测存在误差，SODA依然能够通过指数衰减扰动性质来抑制预测误差的长期影响，保证系统的缓冲区稳定，并且实现较低的动态遗憾和竞争比。

- **鲁棒性**：SODA能在预测不准确的情况下保持较高的QoE，避免重缓冲和频繁切换。
- **性能保证**：通过定理4.2，SODA证明了其在存在误差时仍能达到接近最优的表现，且误差的累积影响是可控的。

这种设计使得SODA不仅在理论上具有优势，也在实践中能够应对网络环境中常见的带宽预测不准确的挑战。



## 4.3 Computational Efficiency

在第 4.3 节“计算效率”（Computational Efficiency）中，作者首先指出：

- **精准求解的不可行性**
   为了每次做出最优的比特率决策，按公式 (2) 对未来 K 个时隙同时进行搜索，其时间复杂度是 O(|R|^K)，呈指数增长。在直播场景下，每 Δt 秒就必须给出下一步决策，这么高的计算量显然无法实时满足播放器的性能要求。 ​
- **单调轨迹近似的关键观察**
   作者给出一个核心洞见：在实践中，当切换代价 γ 足够大时，真正最优的比特率序列往往是要么单调非减，要么单调非增的。换言之，最优解中不会频繁出现“先升后降”再“升”的复杂模式，而是沿着一个方向平滑调整即可近似最优。
- **非正式定理（Theorem 4.3）**
   若在当前时刻 n 的预测窗内，容量预测是常数，即
  $\hat ω_n|_{n−1} = … = \hat ω_{n+K−1}|_{n−1}$
   则存在一个单调（全增或全减）的可行比特率轨迹，它与真正最优轨迹的总代价之差仅为
  $O\bigl(K / \sqrt{γ}\bigr).$
   也就是说，随着切换代价 γ 的增大，单调近似的误差以 1/√γ 的速率迅速下降，理论上保证了仅搜索单调序列就足够得到一个与全局最优非常接近的解。 ​

这一发现为后续的**多项式时间近似解法**奠定了理论基础：只需遍历单调比特率序列（而非全空间枚举），即可将搜索空间从指数级缩减到多项式级别，同时误差有明晰的上界 O(K/√γ)。这样既满足了实时性，又保留了接近最优的性能保障。

# 5. 实施细节（Implementation Details）

- 第五章聚焦于将理论设计转化为实际部署时的具体实现问题，重点解决以下三个核心挑战：
  1. **如何将基于时间的ABR控制器转换为基于分段的架构**（与MPEG-DASH标准兼容）。
  2. **如何鲁棒地整合吞吐量预测**（应对预测误差）。
  3. **如何高效求解预测优化问题**（降低计算复杂度）。

## 5.1 基于分段的架构（Segment-Based Schema）

#### 背景与问题

- **理论设计的矛盾**：SODA在理论上采用**基于时间的优化框架**（将视频视为连续流，按固定时间间隔Δt决策比特率），但实际视频流媒体服务（如MPEG-DASH）要求视频按**分段（Segment）下载**，每个分段包含固定时长的视频内容（如2秒）。
- **核心矛盾**：基于时间的优化需要与分段下载的实际约束兼容。

#### 解决方案

1. **时间间隔与分段对齐**：

   - 将优化阶段的时间间隔Δt设置为与**分段长度相等**（例如Δt=2秒）。
   - **合理性**：在稳定状态下，下载一个分段所需时间通常接近分段长度（或更短），因此时间间隔与分段对齐可减少决策偏差。

2. **防止下载时间过长**：

   - 引入**启发式规则**：控制器选择的比特率不得高于以下值：

     $$r_{selected} \le min \{ r \in R \mid r \ge \hat \omega\}$$

     其中,$\hat \omega$是当前预测的吞吐量。

   - **目的**：避免因选择过高比特率导致单次下载时间显著超过Δt，从而打破基于时间优化的假设。

3. **缓冲区动态管理**：

   - 在基于时间的优化中，缓冲区状态通过时间间隔Δt更新，而实际下载分段时，缓冲区填充量由分段比特率和吞吐量共同决定。
   - 通过对齐Δt与分段长度，确保理论模型中的缓冲区动态（如填充与消耗）与实际分段下载的物理过程一致。

#### 意义

- **兼容性与效率**：在保持理论分析优势的同时，与现有流媒体标准（如MPEG-DASH）无缝集成。
- **稳定性保障**：通过限制比特率选择，避免因网络波动导致的极端情况（如缓冲区下溢或溢出）。

## 5.2 稳健地纳入预测

根据第4.2节的内容，SODA在设计上对预测误差具有鲁棒性，只要预测误差中没有系统性偏差。考虑到现实中的多样化网络条件，我们更倾向于简单的吞吐量预测器，这使得SODA具有较高的可部署性，因为它不依赖于复杂的吞吐量预测器。在实践中，我们观察到随着预测时间范围的增加，预测准确性会下降（见图7）。因此，我们将预测时间范围限制在最多10秒。这也得到了我们在第4.1节中发现的支持，即较长的预测时间范围带来的收益递减。

## 5.3 高效近似求解器

let me give you a brief introduction of how adaptive bitrate streaming works.

To prepare a content for adaptive bitrate streaming we first partition the video stream into small segments say  two seconds each and then encode a video  stream into different resolutions.

so that the Adaptive bitr  controller at the client side can  dynamically select the appropriate video  quality at runtime according to the  varying Network  thoughput

 let's look at a concrete runtime   example, the adapted bitr controller or  ABR controller for short may start by  downloading at the lowest cost say  720p the downloaded segments are placed  in your client buffer and then sent to the display for  rendering. as the network condition gets  better and better the ABR controller may  start downloading at higher and higher  qualities. of course things don't always  work so smoothly, let's say the network  conditions suddenly become worse and the  next segment takes too long to download  before the client buffer is completely  drunk. at this point the client  experience what we called a rebuffering  event. 

you can imagine when you're  watching your favorite show and a  loading screen or black screen pop out  that's superno right? after that the a controller is likely to back back up and downloading lower card is again . from this simple example you may

get some initial sense of what are the

tradeoffs that an AB controller is

trying to balance. there are three major quality of experience components. obviously the users want to watch their videos at higher qualities however they don't want to experience rebuffering events which is super annoying. the third point is a little bit subtle although AB worked by switching bit rate in accordance Network conditions. doing so too much is can actually hurt user experience by itself.

this is because this leads to inconsistent video quality and you can imagine if your picture quality fluctuate all the time in fact by observing the data from a live sports event on Prime video we notice that when the bit trate bit rates switch for more than 20% of the segment a user is likely to abandon before watching even temp % of the stream.

 the problem of ab adaptive bitay streaming has been studied extensively in literature before. there are three groups of designs throughput-based controllers look at the predicted through Network shut and make a bitrary decision.

 buffer based controllers like Bola or BBA look at the current buffer level and choose a bitr accordingly, they tend to be more robust but they also tend to switch bit trades more often.

 the third category hybrid controllers take into account both the predicted thoughput and the current buffer level such designs include MPC which use control series to plan for a look plan for future segments and pensive with which use reinforcement learning and of course there are many other designs.

 given so many choices uh natural question you may ask is why do we need yet another adap controller like soda. I would like to argue that there are several desired properties that we want an AB controller to hold which is not satisfied simultaneously by all by any of the previous controllers. first video providers want to provide consistent high quality video streaming experience to all users regardless of what devices they're using what network connection they are on.

 by consistent I mean less reffering and infrequent switching it turns out the previous design usually sacrific one into favor of the other two QE components. second we want our ABR controller to have theoretic theoretical guarantees for all three QE components.

 the closes we have from the literature is bowler which has thetical guarantees for quality and rebuffering but not bitr switching .

so we want our AVR controller to be robust against super production erors which are inevitable given the diverse nwork conditions in the world .

and finally we want our av controllers to be press Cod for production deployment to maximize its practical impact.

 now is the time to give you a high level overview of how soda works, given a predicted Network thoughtput

soda also try to solve an optimization problem that minimize over the next K time intervals. notice it's time

intervals not next Cas video segments

more on this later later anyway it

minimize over the next K time intervals

the sum of distortion cost buffer cost

and switching costs subject to buffer

Dynamics and constraints the three costs

here correspond to the three QE

components this abstract formulation by

itself is not very useful because it

didn't provide cical guarantees nor

robustness

guarantees to do that we need to

transform the this problem into a socal

problem to take advantage of the recent

advancement in the serial community socal

refers to smoothness

optimized onl online conx optimization

convex

optimization it's essentially a com a

traditional online comx optimization

with switching cost it turns out that to do this transformation

we need to do we need to we need to have

several modeling

Innovations and but just by taking the

traditional formulation of ab will not

work

here I will not bore you with all the

mathematical details here but I do want

you to take away two important two key

model modeling Innovations we

have the first one is we pivot from the

traditional segment based View to a Time

based view of adaptive bitrate

streaming traditionally people formulate

the problem of a adaptive bitrate

streaming in terms of this cre segment

by segment based process that is a

natural formulation given the segment by

segment download nature of dash of Dash

playback however however we choose to

view the video stream as a continuous

fluid stream and forget about the

segment Bund for thetical analysis

purpose this is inspired by how people

analyze TCP performance they also treat

TCP stream as a fluid stream instead of

discrete

packets the second modeling Innovation

we made is transitioning from modeling

rebuffering time directly to modeling

buffer stability

instead because there's the there's a QE

components called rebuffering time

people intuitively try to model

rebuffering time directly as you can see

in the figure shown

here there are two problem with

this one is that even if the buffer

level is dangerously close to zero there

is no penalty whatsoever because the

controller thinks there's no

reffering the second is that this

formulation is more like a hard

constraints which is not directly

optimizable to overcome this we choose

to model buffer stability instead so we

have a Target buffer level and we impose

more and more penalty as the buffer

level drops below the target this way

the optim the objective function become

smooth and the algorithm turn out turns

out to be more robust in

practice with these modeling Innovations

we successfully transformed our AVR

problem to a soal problem and we can say

some nice properties about how a soda

will behave in

practice in

particular we can theoretically shown

that soda satisfied exponentially

decaying privation

property let me work you through let me

let me help you understand this by

working hisory and

illustration let's say the black line

here is the optimal trajectory of an AVR

controller each of these labels are

State you can think of it as the buffer

level and the

bit rate at each point when Soda makes a

decision it may make a mistake because

of PR super prediction errors or other

factors which is

inevitable how what's nicest about soda

is that we can theoratically proved that the

impact about impact of each mistake WIll

Decay exponentially into the

future in other words the effects of the

mistakes will never

accumulate this is in contrast to other

systems where one mistake may cause

cascading C cascading effect in the

future as you can

see soda the the trajectory of soda were

not deviate from the optimal trajectory

too much in fact we have a theorem stating

that the cost of the so the cost the

total cost incurred by soda divided by

the total cost incurred by the offline

optimal algorithm is 1 plus big O rho

to the K where rho is the factor smaller

than one and K is the look at window

lens now that a ratio of one essentially

means an algorithm is optimal

here is a exponential decaying factor

meaning that the performance of soda

will convert to the offline optimal as

the loal window increase exponentially

fast in fact we have another serum

precisely quantifying the performance of

soda in relationship to the magnitude of

the prediction era but that is a bit

more technical so I will refer you to

our paper for more technical

details





now it's what's great about soda is just that not only does it have like sound

thetical guarantees and interesting

theoretical properties but also it does

perform well in practice we first

Implement a prototype in browser and

conducted conducted emulations on puffer

which is an open uh AVR research

platform We compare soda with a bunch of

state-of-the-art

baselines what we find that is soda

achieves the higher C highest overall

card of experience by a significant

margin compared to other

baselines in particular it does so by

achieving a reasonable

quality a very low reffering

ratio and very low switching rates

 inspired by the great performance we from the Prototype evaluation we decided

to try it out in production but before

that we need to make sort of practical

for it to able to run in the world by

practical I specifically mean it has to

run fast given there are some very

lowend devices in a production

environment. so soda has to be

efficient we found that soda is trying to

solve an optimization problem over the

next few time intervals

so to do that you the naive thing to do

is you enumerate over all combinations

which is their exponential amount of

them so brute force or a solver uh just

a commercial solver is obviously a nogo

in

practice some researchers propose table

looka before but that is not scalable

and will incur High operation

overhead now through theoritical analysis we

discover that for soda we only need to

consider monotonic betray sequence a

monotonic I mean it's either increasing

like this or

decreasing this translate to only less

than 150 iterations in practice which is

not a big deal even for lowend

devices

 next we take soda to the production Network on Prime video we try it out for users in more than 10

countries and for some major for some

large live event sports

events the interesting thing I want you

to notice here is that soda reduces the

bitrate switching come by almost 80%

compared to a well tuned production Baseline

think about it it's 80% a huge and

dramatic Improvement in terms of user

experience and what's more important is

that it does so without sacrificing the

other quality of experience components

it has slightly higher mean bit rate and

it has lower reffering

ratio another takeaway from the

production deployment is that we soda ex

can successfully run even on lowend

devices like set-top

boxes and soda achieves this amazing

performance by without relying on any

sophisticated super predicts

in fact we just use the simple sliding

window predictor and that's

it 

finally to summarize with time base and buffer stability modeling we can

provide theoretical guarantees for soda

using smoothness optimized convex

optimization

framework soda is highly performance

robust against prediction eror and

practical

the we believe the theortical Insight

behind Soder in particular the

exponential decay peration property can

be potentially transferred to other

networking domains in fact as you may

imagine the exponentially decaying

pration property is a very nice property

to have from even for other network

systems meaning the errors the effects

of Errors do not accumulate into the

future that concludes my presentation

thank you very much

