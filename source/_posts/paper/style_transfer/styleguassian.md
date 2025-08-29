---
title:  "StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting论文理解"

date:  2025-4-14 11:28:00

tags:  [论文,风格迁移]

categories:  [风格迁移]

comment:  false

toc:  true
---


#

<!--more-->

- SIGGRAPH Asia 2024
- [代码](https://github.com/Kunhao-Liu/StyleGaussian?tab=readme-ov-file)
- [论文](https://arxiv.org/abs/2403.07807)

# 0. Abstract

- 3D风格迁移
- 即时编辑
- 利用3D高斯飞溅（3DGS, 3D Gaussian Splatting），StyleGaussian在不影响其实时渲染能力和多视图一致性的情况下实现了风格迁移。
- 它通过嵌入、迁移和解码三个步骤实现即时风格转移。最初，2D VGG场景特征被嵌入到重建的3D高斯中。接下来，根据参考样式图像变换嵌入的特征。最后，将变换后的特征解码为风格化的RGB。
- StyleGaussian有两个新颖的设计。
  - 第一种是高效的特征渲染策略，首先渲染低维特征，然后将其映射到高维特征，同时嵌入VGG特征。它显著降低了内存消耗，并使3DGS能够渲染高维内存密集型特征。
  - 第二种是基于K-最近邻的3D CNN。作为风格化特征的解码器，它消除了损害严格多视图一致性的2D CNN操作。





## 0 背景知识

### 0.1 神经辐射场（Neural Radiance Field, NeRF）

**NeRF概念：** 神经辐射场是一种先进的3D场景表示方法，可从一组多视角2D图像重建出逼真的3D场景。NeRF使用一个5D坐标（空间位置 (x,y,z) 加视角方向）的函数，经由神经网络（通常是MLP）映射到颜色和密度。渲染时，通过在摄像机视线（光线）上采样多个3D点，累积这些点的颜色和密度，得到最终像素颜色。NeRF能够建模视角相关的效果（如反射高光），生成极为真实的视图。

**NeRF的缺点：** NeRF的场景信息隐含地存储在神经网络权重中，这种**隐式表示**使得直接编辑场景（例如更改风格或外观）变得困难。与显式3D表示（如网格或点云）不同，编辑NeRF通常需要调整神经网络参数，这涉及耗时的优化过程。此外，NeRF渲染速度相对较慢，因为每个像素都需要大量采样和MLP计算。为提高重建质量和速度，已有各种NeRF变种，例如使用体素网格或混合表示等。

### 0.2 3D Gaussian Splatting（3D高斯铺撒）

**概念：** 3D高斯铺撒（3DGS）是一种显式的辐射场表示方法，它使用大量**三维高斯分布**来表示场景中的颜色与密度。每个高斯可以理解为一个“模糊的”三维点，具有中心位置（均值）、形状大小（协方差矩阵）、不透明度以及颜色参数。渲染时，并不采用NeRF那样逐点采样光线，而是将这些高斯分布直接投影到图像平面进行**光栅化**（splatting），按照深度顺序将覆盖同一像素的多个高斯的颜色混合。像素颜色计算公式如下：

C=∑i∈Nciαi∏j=1i−1(1−αj),C = \sum_{i \in N} c_i \alpha_i \prod_{j=1}^{i-1}(1-\alpha_j),

![image-20250414203019098](../../../../../../AppData/Roaming/Typora/typora-user-images/image-20250414203019098.png)

其中每个高斯 i 对应的$\alpha_i$由其不透明度和在该像素处的覆盖范围共同决定。简单来说，就是**按顺序将多个高斯“铺撒”到像素上进行混合**。

**优点：** 与依赖体积采样的NeRF渲染相比，3DGS采用光栅化方式渲染，大幅提高了效率。3DGS可以在**实时**速度下渲染，同时重建质量优秀，在近期受到广泛关注。例如，Kerbl等人在2023年的研究表明，3DGS具备快速重建能力、实时渲染性能以及良好的重建结果。3DGS之所以能够实时，是因为它利用GPU对点云进行并行光栅化，而避免了NeRF中耗时的逐像素采样和神经网络计算。总的来说，3DGS提供了一种相比NeRF更加高效的3D表示和渲染方案。

**不足：** 由于3DGS使用离散的高斯分布来表示场景，有时在极精细的细节上可能有所欠缺。另外，与NeRF一样，标准的3DGS主要关注颜色/密度的表示与渲染，本身并没有提供直接的高层语义编辑接口。

### 0.3 风格迁移基础



**3D风格迁移：** 风格迁移也扩展到了三维领域。一些工作尝试对3D数据如点云或网格进行风格化。但由于3D数据形式与2D图像差异大，这些方法往往需要复杂的处理，而且渲染性能往往不及辐射场方法。一些最新研究开始探索**辐射场的风格化**。部分方法采用每次优化来调整NeRF的参数从而实现风格迁移，效果很惊艳但每次换风格都得耗费很长时间优化。也有方法训练**通用模型**实现任意风格的一次性迁移，例如Chiang等人的HyperNeRF/HyperNet方法通过将风格嵌入到NeRF网络权重中，实现任意风格迁移。不过这些方法渲染速度较慢且在细节纹理上表现不足。还有StyleRF方法实现了**零样本（zero-shot）**的辐射场风格迁移，但依赖2D卷积解码，导致多视角一致性受到破坏。为了解决上述问题，StyleGaussian应运而生。

### 0.4 VGG网络与特征提取

**VGG网络简介：** VGG是一系列卷积神经网络模型（如VGG-16）。这些网络在ImageNet等数据集上预训练，用于图像分类，但它们的中间层特征在风格迁移等任务中非常有用。VGG网络由多个卷积层和池化层堆叠而成，特征层通常用名称如`Conv4_1`或`ReLU4_1`等表示（即第4块卷积后的激活）。**较浅层**特征保留了更多的局部纹理细节，**较深层**特征则提取了更抽象的内容结构。

**特征提取：** 给定一张输入图像，我们可以将其输入预训练的VGG网络，在指定的层处截断网络，输出该层的特征映射。例如论文中提到的`ReLU3_1`层输出特征维度为256，`ReLU4_1`层输出维度为512。这些高维特征可以被看作图像在该层的表示，每个通道包含图像某种模式的响应。风格迁移中，经常会提取多层的特征来分别衡量内容和风格。例如内容损失常用较深层特征的MSE差异，风格损失则用多个层特征的Gram矩阵差异。

**在3D场景中的应用：** 在这篇论文中，作者将预训练的VGG用于**提取多视角图像的特征**，再把这些特征“嵌入”到3D高斯上。这需要一种策略将2D图像特征和3D点云对应起来，下文会详细介绍。

### 0.5 多视图一致性

**定义：** 多视图一致性指的是：当我们从不同视角查看同一个3D场景时，所看到的风格化结果在空间上应保持一致，不应出现视角变化导致的风格失真或闪烁。简单来说，一个物体的某个细节在不同角度下看应该有相同的风格纹理，而不是因为从新视角渲染又重新生成了不同的纹理。

**为何重要：** 在2D风格迁移中，每张图片独立处理没有此问题。但在3D风格迁移中，如果采用逐视图的方式对每帧进行风格化（例如对NeRF渲染出的每个视图用2D网络风格化），可能各帧之间缺乏约束，导致出现视角不一致的情况，比如相邻帧物体表面的花纹对不齐、颜色跳变等。这在观看连续的视角变化（如视频或VR交互时）会导致明显的闪烁和割裂感，破坏沉浸体验。

**解决思路：** 为保证一致性，需要在**3D表示层面**进行风格迁移，而非单纯在2D图像后处理。也就是说，风格应该附着在3D物体上，再从各视角看到的都是同一个风格化的3D物体。StyleGaussian正是遵循这个原则：它直接在3D高斯上操作颜色和特征，使得风格迁移发生在三维空间中，并采用严格与视图无关的解码方式，避免了任何逐帧的不一致处理。另外，多视角一致性的评估在论文中通过光流warp度量来实现，反映不同视角下对应区域的差异（见论文实验部分）。

### 0.6 3D CNN 与 KNN卷积

**3D CNN的概念：** 卷积神经网络（CNN）在2D图像上通过在规则网格上滑动卷积核提取局部模式特征。然而，3D点云（如高斯点的集合）并不是规则的网格结构。因此，需要一种针对非规则点集合的“卷积”操作。3D CNN在本文中特指**作用于点云或3D空间邻域的卷积运算**。常见做法是将每个点的**K近邻（K-Nearest Neighbors, KNN）**看作局部邻域，相当于在点云上定义了一个“感受野”。

**KNN卷积原理：** 对于每个中心点，我们找到在3D空间离它最近的K个点（包括它自身）作为邻域。然后类似2D卷积，将这个邻域的特征拼在一起，通过一个小型神经网络（具有可学习的权重和偏置）计算输出特征，赋给中心点。这一过程可以视为卷积核权重作用在邻域点特征上的加权求和，再加偏置，可能还跟随激活函数。由于点云没有固定顺序，需要卷积操作对不同排列保持不变，PointNet等方法用对称函数（如最大值或平均值）来聚合邻域特征。而本文采用的是直接对邻域的**所有K个点特征应用线性变换并求和**，本质也是一种卷积形式。

**实现机制：** 作者将这种KNN卷积实现为矩阵乘法以充分利用GPU并行。具体来说，将所有点的K邻域特征堆叠成大矩阵，将卷积核权重也展成矩阵，一次矩阵乘法即可并行计算出所有点卷积后的结果。通过多层堆叠，这个3D CNN能够逐步扩大每个点的感受野，捕捉更大范围的风格模式。

**与其它解码方式对比：** 论文在消融实验中比较了不同的解码器设计：纯MLP逐点解码、PointNet结构解码和本文的KNN卷积解码。结果表明：PointNet由于对称函数聚合，使得一组邻域点输出相同颜色，缺乏变化，导致每个高斯几乎颜色相同；MLP逐点解码有一定风格但局限于点自身，无法呈现连续的笔触纹理；而KNN卷积利用邻域信息，既保证了局部连续性又带来多样的色彩变化，能表达更复杂的风格纹理。这说明KNN卷积的解码器在捕捉风格模式和保持3D一致性上效果最好。

### 0.7 高维特征嵌入及优化

**问题由来：** 为了将2D图像的风格应用到3D场景，我们需要让3D场景“知道”自身的内容特征。作者的做法是将每个3D高斯赋予一个**高维特征向量**，使其能够表示VGG提取的内容特征。但是VGG特征维度很高，例如使用VGG的`ReLU4_1`层有512维。如果直接给每个高斯存512维特征来渲染，将非常耗内存和计算。论文中指出，即使降到256维，在他们的GPU（24GB内存）上编译渲染核仍然失败，因为共享内存不够。

**低维嵌入策略：** 为解决这个内存瓶颈，作者提出了**高效特征渲染策略**：先让每个高斯学习一个低维特征（论文中设定维度D′=32）。渲染阶段，像渲染颜色一样渲染出每个像素的低维特征图 F' ([2403.07807v1.pdf](file://xn--file-erb5mprrfzbqlmyrk8cg52%23:~:text=dering low, rd-3u19ay86d/))。随后，通过一个可学习的**仿射映射**（线性变换+偏置）将低维特征F'映射到高维特征空间 F。换句话说，网络学得一个矩阵T和偏置，将32维特征变换成512维的VGG特征。由于线性变换可以在渲染求和后与特征求和交换顺序，这一映射同样适用于像素累积的结果。这样，每个高斯最终还是拥有了对应VGG高维特征，只是这个过程被拆分为“先低维渲染，再升维”的两步，从而**避免了直接渲染高维特征的内存开销**。

**特征优化：** 将特征嵌入到高斯后，作者通过一个**蒸馏（distill）**过程来优化这些特征值。具体来说，他们用多视角图片（训练图像）经过VGG得到的特征图作为“真实特征$F^{gt}$，然后调整高斯的特征使得用高斯渲染得到的特征图 F 与 $F^{gt}$ 尽可能接近。损失函数为两者的L1范数差异。通过梯度下降优化每个高斯的特征参数$f_p$，直到3D高斯渲染的特征能够重现原始多视图的VGG特征。完成这一优化后，每个高斯都携带了一个反映场景内容的512维特征向量。这一步相当于预先将场景的内容信息嵌入3D表示。作者在风格迁移过程中将这些特征固定不变，表示内容结构保持不变。





### 1. 引言

引言部分首先介绍了研究背景和挑战，然后简述作者的方案和贡献。

**1. 背景 - 3D辐射场的兴起与编辑难题：** 随着神经辐射场技术的发展，3D场景重建取得了重大进展，使人们能够沉浸式地探索3D世界。辐射场将3D坐标映射到颜色和密度，实现新视角合成，在计算机视觉和图形学中有广泛应用。常见实现包括隐式的MLP NeRF、显式的体素网格，或两者结合的混合表示。**然而**，与传统显式表示（网格、点云）相比，辐射场**难以直接编辑**其风格和外观。因为辐射场的信息隐含在网络权重或大规模张量中，很难像编辑纹理贴图那样直观地修改。以往一些工作尝试在图像、文本或其它用户输入的引导下，通过学习的方法编辑辐射场，但通常需要**测试时的优化**（test-time optimization），每进行一次编辑都要耗费大量时间调整参数。另一些方法训练前馈网络加速编辑，但提升有限而且可能破坏多视角一致性。总而言之，实现既快速又一致的3D风格编辑仍是挑战。

**2. 实时3D风格迁移的需求：** 引言接着指出，为了3D交互的流畅体验，3D风格迁移必须能够**瞬时完成**，并且不能牺牲渲染的实时性和多视角一致性。最近兴起的3D高斯铺撒（3DGS）由于渲染快速，被认为有望支撑**即时的3D风格迁移**。当前最先进的3D风格迁移方法一般包含三个步骤：（1）**特征嵌入**：将2D图像的VGG特征嵌入到重建的3D几何上；（2）**风格迁移**：根据风格图像转换嵌入的特征；（3）**特征解码**：将风格化的特征解码回RGB图像。尽管3DGS可以有效地渲染RGB（三维）图像，但渲染高维特征并将其嵌入到重建的3D几何图形中是计算和内存密集型的。此外，大多数已有方法在第（3）步使用2D CNN来解码特征到图像。这种做法往往会破坏严格的多视角一致性，并降低3D风格迁移的质量。也就是说，如果用2D卷积网络对每张视图进行处理，不同视角下可能出现风格不连贯的现象。

**3. 作者的方法和贡献：** 基于上述考虑，作者设计了StyleGaussian，一种实现**即时3D风格迁移且严格保持多视角一致性**的管线。StyleGaussian包含前述的三个步骤：首先将图像的VGG卷积特征嵌入3D高斯（内容嵌入）；然后输入任意风格图像，将3D特征分布调整为该风格（风格变换）；最后用3D卷积解码得到彩色的高斯点（特征解码）。如此，用户只需给出一张风格图，系统便可立刻产出整个3D场景的风格化结果，**不需要针对新风格进行任何优化**（zero-shot）。引言最后列出了本文的主要贡献：

- 首先，提出了StyleGaussian这一新的3D风格迁移流程，实现了**10fps的即时风格迁移**，同时保留了3DGS的实时渲染能力和严格的多视角一致性。
- 其次，设计了**高效的特征渲染策略**，可以在学习嵌入高维VGG特征的同时仅渲染低维特征，从而成功在3DGS框架下渲染高维特征。这一策略大幅减少了内存占用，使高维特征嵌入成为可能（这是实现高质量风格迁移的基础）。
- 第三，开发了一种**KNN卷积的3D解码网络**，作为特征到RGB的解码器。该解码器具有大的感受野，避免了使用2D CNN，从而**严格保证了多视角的一致性**。

通过以上创新，StyleGaussian实现了当前3D风格迁移领域前所未有的性能：**无需对新风格优化，实时交互速度，视角一致且风格质量优秀**。



# 2. Related Work

## 2.1 Radiance Fields

- 辐射场[NeRF]最近在推进3D场景表示方面取得了重大进展。这些场是将辐射度（颜色）和密度值分配给任意3D坐标的函数。像素的颜色是通过体渲染聚合3D点的辐射度来渲染的[26]。辐射场在视觉和图形的各个领域都有广泛的应用，尤其是在视图合成【2，3，35】、生成模型【4，13，32，40】和表面重建【47，48】中。它们可以通过多种方法实现，如MLP[2，3，27]、分解张量[4，5，9]、哈希表[30]和体素[10，43]，许多研究旨在增强它们的质量[2,3]或渲染和重建速度[5,10,11,30,39]。在这些进步中，3D高斯溅射（3DGS）【19】因其快速重建能力、实时渲染性能和出色的重建结果而脱颖而出。它使用大量显式参数化的3D高斯对辐射场进行建模。其实时渲染能力的基石是它依赖于光栅化而不是光线跟踪来渲染图像。基于3DGS的优势，我们的工作利用它来促进身临其境的3D编辑体验。

## 2.2 3D Appearance Editing

- 编辑网格或点云等传统3D表示的外观通常很简单，因为网格与UV maps相关联，而点对应于图像中的像素。然而，编辑辐射场是具有挑战性的，因为它们在神经网络或张量的参数内隐式编码。因此，先前的研究求助于基于学习的方法来编辑辐射场【1，6，7，14，23–25，34，42，45，46，50，52，57】，由图像【1，7，24，53】、文本【6，14，45，46，57】或其他形式的用户输入【6，25，50】引导，包括诸如变形【34，50，52】、外观变化【6，14，24，45，46，57】、移除【6】、重新照明【42】和修复【23，28】的修改。然而，这些方法中的大多数依赖于测试时间优化策略[6，14，28，45，46，53，57]，需要对每次编辑进行耗时的优化过程。或者，一些方法有助于以前馈方式编辑3D场景[7,24]。然而，这些方法的编辑速度仍然远远达不到交互速度。相比之下，我们的方法可以即时编辑场景的外观。

## 2.3 Neural Style Transfer

- 神经风格转移旨在渲染一个新的图像，将一个图像的内容结构与另一个图像的风格模式融合在一起。先前的研究表明，VGG特征的二阶统计量【41】封装了2D图像的风格信息【12】。最初，该领域依赖于优化方法来对齐风格图像的VGG特征【12】，但随后的方法引入了前馈网络来近似这一优化过程【16,18，22】，显著提高了风格转移的速度。最近的努力通过尝试对点云或网格进行风格化，将风格转移扩展到了3D领域【15,29】。然而，与辐射场相比，这些方法在渲染能力上通常滞后，这促使对辐射场的风格化进行进一步研究【7,8,17,24,31,46,53】。**[31，46，53]等作品已经成功地通过优化实现了辐射场风格转移，提供了视觉上令人印象深刻的风格化，但代价是对每种新风格进行耗时的优化，并且对看不见的风格的泛化性有限。**HyperNet[7]等替代方案将样式信息嵌入MLP参数中，用于任意样式传输，但面临渲染缓慢和样式模式细节差的问题。StyleRF[24]引入了零镜头辐射场风格转移，但使用了2D CNN解码器，损害了多视图一致性。然而，我们的方法允许即时传输和实时渲染，同时保持严格的多视图一致性。

### 3. 方法

在方法部分，作者首先介绍了3D Gaussian Splatting的预备知识，然后详细阐述StyleGaussian的三个阶段：特征嵌入（Embedding）、风格变换（Style Transfer）和RGB解码（Decoding）。为方便理解，下面 ([StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://kunhao-liu.github.io/StyleGaussian/))给出了StyleGaussian整体流程的示意图：左侧是原始重建的有色高斯点（内容场景），中间通过嵌入特征和风格转换得到风格化的高斯点云（颜色尚未解码为最终RGB），右侧是最终解码渲染出的风格化结果。

([StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://kunhao-liu.github.io/StyleGaussian/)) *StyleGaussian 方法流程概览 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=Image%3A Refer to caption Figure,as the decoder in))。给定重建好的3D高斯表示 GG（左一），首先进行**特征嵌入** ee：将预训练VGG网络提取的内容特征映射到每个高斯上，得到附有特征的高斯 GeG^e（左二）。然后给定一张风格图像，将嵌入特征进行**风格变换** tt，注入风格信息，获得风格特征高斯 GtG^t（右二，中间偏左）。最后，通过**解码** dd步骤，将每个高斯的风格特征解码为RGB颜色，生成风格化的3D高斯 GsG^s（右一）。其中，作者设计了高效的特征渲染策略用于 ee 阶段渲染高维特征，并开发了基于KNN的3D卷积网络用于 dd 阶段直接在3D上解码颜色，从而保证多视角一致性。*

#### 预备：3D高斯铺撒表示

在正式讲方法之前，作者简要介绍了3D Gaussian Splatting的表示形式（本文的算法就建立在这种表示之上）。我们在背景“3D Gaussian Splatting”一节已经解释了大部分概念，这里再简明回顾： ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=3D Gaussian Splatting ,ordered Gaussians overlapping the pixel))定义一个3D场景由一组高斯原语 G={gp}PG = \{g_p\}_P 组成，每个高斯 gpg_p 参数包括中心 μp\mu_p、协方差矩阵 Σp\Sigma_p（控制形状大小）、不透明度 σp\sigma_p 和颜色 cpc_p ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=3D Gaussian Splatting ,ordered Gaussians overlapping the pixel))。渲染时，对每个像素，将视线穿过像素遇到的高斯按深度远近排序，并按公式将它们的颜色和不透明度累积混合，得到该像素最终颜色 ([2403.07807v1.pdf](file://file-erb5mprrfzbqlmyrk8cg52%23:~:text=specifying its shape and size,,ordered gaussians overlapping the pixel/)) ([2403.07807v1.pdf](file://xn--file-erb5mprrfzbqlmyrk8cg52%23:~:text=i1 j=1-ww34ace/))。这样的高斯渲染器是可微分的，且因采用光栅化方式非常高效，可实现实时渲染 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=))。与NeRF等需要沿光线采样大量点并用MLP计算的方法相比，高斯铺撒在时间和内存上都更高效 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=))。此外，与NeRF需要逐场景训练类似，3D高斯也可以通过带监督的优化从多张带位姿的图像中重建得到 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=approaches used to render radiance,supervised with multiple posed images))。

有了这个3D高斯表示 GG 后，StyleGaussian的任务就是在保持高斯几何参数不变的前提下，改变每个高斯的颜色 cpc_p 使其具有目标风格 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=We present StyleGaussian%2C a 3D,view consistency))。这一点很重要：**几何结构固定不动，仅调整颜色**，以保证场景内容不变 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=3D Gaussians of a scene%2C,view consistency)) ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=))。由于渲染过程与原始高斯表示相同，风格化后的3D高斯仍可实时渲染，并天然保持视角一致性 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=3D Gaussians of a scene%2C,view consistency))。

#### 4.1 特征嵌入（Feature Embedding）

**目的：** 将2D图像的深度特征嵌入3D场景，使每个3D高斯都携带内容表示。直观来说，就是让点云“记住”原始场景的视觉内容，这样在风格转换时才能区分哪里该保留内容结构、哪些部分应用风格纹理 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=We denote the style transfer,VGG image features into))。

**过程：** 给定一个已经重建好的场景高斯集合 G={gp}G=\{g_p\}（具有固定的几何和初始颜色），我们为每个高斯分配一个**可学习的特征向量** fpf_p ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=Specifically%2C given the reconstructed 3D,feature of the same Gaussian))。作者使用预训练的VGG网络提取输入图像（场景的某视角）的中间层特征映射作为“真值”特征 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=The VGG features of a,the feature embedding process below))。然后，让这些高斯通过渲染器直接渲染“特征图”。具体而言，类似于前述像素颜色的累积公式，每个像素的特征 F(x)F(x) 可渲染为穿过该像素射线的各高斯特征的加权和：

F(x)=∑i∈N(x)wi fi,F(x) = \sum_{i \in N(x)} w_{i} \, f_{i},

其中 N(x)N(x) 表示射线经过的高斯集合，wiw_i 是高斯 ii 对该像素的混合权重（与之前的不透明度累积公式类似，由α\alpha计算得到） ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=Specifically%2C given the reconstructed 3D,feature of the same Gaussian))。这样，对于每张训练图像，我们都可以渲染出对应视角下高斯的特征图 FF。与此同时，我们也能从原始图像经过VGG提取得到真实的特征图 FgtF^{gt}。之后，通过**优化高斯的特征向量 {fp}\{f_p\}**，使渲染特征图 FF 尽量逼近 FgtF^{gt}。损失函数是两者的逐像素差的L1范数：

\mathcal{L}_{\text{embed}} = \sum_{x} \| F(x) - F^{gt}(x) \|_1  ([[2403.07807] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=)).

不断迭代后，每个高斯的特征 fpf_p 都学到了能重现对应图像VGG特征的值。

**高维特征降维映射：** 但是，如前所述，每个高斯特征维度D很高（例如512维），直接这样渲染计算开销巨大 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=However%2C as VGG features have,is now %2C and the))。作者发现，当D=256时渲染器甚至因为显存限制无法编译 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=through Gaussian Splatting is both,can be rendered as))。为此，他们引入了**低维-高维映射策略** ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=most consumer GPUs,can be rendered as))。具体做法是：将每个高斯的特征向量由 fp∈RDf_p \in \mathbb{R}^D 替换为一个低维向量 fp′∈RD′f'_p \in \mathbb{R}^{D'}（论文设定 D′=32D'=32），先渲染出低维特征图 F′∈RD′F' \in \mathbb{R}^{D'} ([2403.07807v1.pdf](file://xn--file-erb5mprrfzbqlmyrk8cg52%23:~:text=dering low, rd-3u19ay86d/))。然后设计一个仿射变换 T:RD′→RDT: \mathbb{R}^{D'} \to \mathbb{R}^{D}，在特征渲染之后再把 F′F' 投影回高维空间 ([2403.07807v1.pdf](file://xn--file-erb5mprrfzbqlmyrk8cg52%23:~:text=first render the low,f -8y77b/))。形式上，对于每个高斯：

fp=T(fp′)=Wfp′+b,f_p = T(f'_p) = W f'_p + b,

其中WW是可学习的线性映射矩阵，bb是偏置 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=))。由于渲染过程对每个像素实际上是线性累积（权重和特征的乘加），这种先映射再累积与先累积再映射可以证明是等价的 ([2403.07807v1.pdf](file://xn--file-erb5mprrfzbqlmyrk8cg52%23:~:text=and then map them to,formation applied to f -3296d/)) ([2403.07807v1.pdf](file://file-erb5mprrfzbqlmyrk8cg52%23:~:text=i.xn-- t can be refor,f i from f -yq50a/))。因此，我们可以先用低维特征渲染得到 F′F'，再对 F′F' 的每个像素应用 WW 和 bb 得到对应的高维特征图 FF ([2403.07807v1.pdf](file://xn--file-erb5mprrfzbqlmyrk8cg52%23:~:text=first render the low,f -8y77b/)) ([2403.07807v1.pdf](file://file-erb5mprrfzbqlmyrk8cg52%23:~:text=i.xn-- t can be refor,f i from f -yq50a/))。这样就成功绕过了直接渲染512维特征的问题。在训练优化时，WW和bb也是参与优化的参数，使得最终 F=WF′+bF = W F' + b 能够匹配 VGG 的高维特征图 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=))。

**结果：** 通过上述过程，作者成功地为3D高斯场景嵌入了与多视角图像**一致的VGG特征** ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=where donotes the affine transformation,the speed of style transfer))。这些特征一旦优化好，就固定在每个高斯上，不再变化 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=))。至此，场景的**内容表示**已经注入3D表示中，为下一步的风格转换打下基础。值得一提的是，这个特征嵌入过程是在离线进行的（预先训练阶段），因此不影响实际风格迁移时的速度 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=))。

#### 4.2 风格迁移（Style Transfer）

在每个高斯都带有内容特征后，接下来就是根据输入的风格图像对这些特征进行变换。作者选择了前面介绍的**AdaIN算法**来实现零时延的风格注入 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=After obtaining the 3D Gaussians,a Gaussian is given by))。具体做法是：先用同样的VGG网络提取风格参考图像 IsI^s 在相应层的特征图 FsF^s。对每个高斯 gpg_p，它有嵌入的内容特征 fpf_p（维度D）。我们将所有高斯的特征集合视为内容特征分布，将其整体的均值 μ(f)\mu(f) 和标准差 σ(f)\sigma(f) 计算出来 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=))。然后对每个高斯的特征执行AdaIN变换：

fpt=AdaIN(fp,Fs)=σ(Fs)(fp−μ(f)σ(f))+μ(Fs),f^t_p = \text{AdaIN}(f_p, F^s) = \sigma(F^s)\left(\frac{f_p - \mu(f)}{\sigma(f)}\right) + \mu(F^s),

其中 FsF^s 表示风格图像的VGG特征，μ(Fs)\mu(F^s)和σ(Fs)\sigma(F^s)是其通道均值和标准差 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=))。经过这样处理后，得到**变换后的高斯 GtG^t**，每个高斯的特征 fptf^t_p 都已经带有风格图像的统计特征，相当于内容特征被**风格化**了 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=))。这一过程不需要训练任何额外参数，计算非常迅速，可在实时交互中即时对新风格生效 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=After obtaining the 3D Gaussians,a Gaussian is given by))。

需要注意的是，此时高斯的特征虽然已经是风格化特征，但还不是最终的颜色值。下一步我们还需将这些特征解码成RGB颜色。

#### 4.3 RGB解码（RGB Decoding）

**挑战：** 完成风格特征的计算后，要得到最终的彩色图像，有两种思路：一种是先把每个视角的风格特征图像渲染出来，再用2D网络把特征图转成风格化图像（此前一些方法如StyleRF即采用此策略） ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=process ,over an MLP for decoding))；另一种则是**直接在3D空间将特征变为颜色**，也就是给每个高斯赋予一个RGB值，然后按正常方式渲染出图像 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=process ,over an MLP for decoding))。作者选择了后者，即在3D点云上直接解码，以保证渲染流程不变，从而不破坏实时性和多视角一致性 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=process ,over an MLP for decoding))。

如果直接为每个高斯训练一个MLP将其特征映射到颜色，其感受野局限在单一点，难以表达复杂的风格花纹 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=ensuring real,MLP for decoding the features))。正如之前所述，**风格化的颜色往往需要“大范围的感受野”\**才能准确解码 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=ensuring real,MLP for decoding the features))。这也是之前很多研究偏好用2D CNN而非MLP进行解码的原因，因为2D CNN可以在图像平面上看见邻近区域，从而生成连贯的纹理 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=ensuring real,MLP for decoding the features))。为解决这一问题，作者设计了一个\**简单而有效的3D卷积解码器** ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=We design a simple and,For each layer%2C we refer))。

**KNN卷积解码器结构：** 这个解码器在3D高斯点云上进行卷积运算。具体来说，对于每个高斯，我们找出其在空间中的K个最近邻高斯（包括它自身）构成邻域 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=layers%2C as shown in Fig,Consequently%2C the convolution operation))。这个邻域在卷积中扮演类似2D卷积核覆盖的感受野窗口 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=Image%3A Refer to caption Figure,Gaussians in the next layer))。解码器可以有多层卷积，逐层更新特征 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=We design a simple and,For each layer%2C we refer))。最终一层输出通道数为3，即得到每个高斯的RGB颜色 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=transformed features to the stylized,The parameters))。以单层卷积为例，假设卷积核权重为WW（维度 3×K×D3 \times K \times D，3是输出通道RGB，DD是输入特征维度），偏置为bb（维度3）。对于中心高斯 gpg_p 及其邻域{gk}K\{g_{k}\}_{K}，卷积计算可以表示为：

op=σ(W⋅[fpt∥fk1t∥...∥fkKt]+b),o_p = \sigma\Big( W \cdot [f_{p}^t \| f_{k1}^t \| ... \| f_{kK}^t] + b \Big),

其中 [⋅∥⋅][ \cdot \| \cdot ] 表示将邻域各点的特征级联成一个长向量，σ\sigma是激活函数（文中使用Sigmoid） ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=sliding window%2C assigned to the,at can be expressed as))。这个输出 opo_p 就是中心高斯 gpg_p 的更新特征。多层卷积堆叠后，最后一层将输出维度设为3，每个高斯得到最终RGB值 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=transformed features to the stylized,The parameters))。

作者将上述3D卷积高效实现为矩阵乘法：将所有高斯邻域的输入特征堆叠成一个大矩阵，将卷积权重也展平成矩阵，然后一次矩阵相乘就可并行算出所有输出 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=where denotes the Sigmoid function,9 as))。这种实现等价于逐点滑动邻域卷积，但充分利用了GPU的并行能力 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=where denotes the Sigmoid function,9 as))。

**解码效果：** 通过KNN卷积解码，所有高斯都被赋予了新的颜色，即**风格化后的RGB** ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=transformed features to the stylized,The parameters))。此时，我们获得了风格化的3D高斯集合 Gs={gps}G^s = \{g_p^s\}。渲染这些高斯就能得到各视角下的风格化图像。由于颜色是基于3D邻域一致计算的，每个高斯的颜色在整个过程中唯一确定，不同视角下看到的同一高斯当然颜色相同，从而保证了视角一致性。

另外值得强调的是，风格迁移过程对每个风格图像只需执行**一次** ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=easier to implement to be,preservation and the stylization effect))。也就是说，一旦将场景高斯 GG 转换为 GsG^s，之后无论从什么视角渲染，都不需要重新计算风格化——直接用标准的高斯渲染即可实时得到结果 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=while maintaining strict multi,preservation and the stylization effect))。这与一些需要“边渲染边风格化”的方法形成对比：后者可能每改变视角都要重新经过风格网络处理，既慢又可能不一致 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=reference image,Quantitatively))。StyleGaussian成功将风格迁移与渲染解耦：**先在3D层完成风格化，后在渲染层即时呈现**。

**训练细节：** 作者在训练StyleGaussian的解码器时，采用了与AdaIN论文类似的损失 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=are able to render stylized,preservation and the stylization effect))。具体包括：内容损失（风格化结果与原内容在VGG特征上的MSE），风格损失（风格化结果与目标风格在通道均值和标准差上的MSE） ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=are able to render stylized,preservation and the stylization effect))。二者的加权和作为总损失，其中权重参数可以调节内容与风格的平衡 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=are able to render stylized,preservation and the stylization effect))。通过这样的训练，多视角的一组风格图像共同优化解码器的参数，使其能够在保持内容的同时充分应用风格纹理。有关内容/风格平衡的具体效果，论文在消融实验中也有所讨论。

### 实验

作者通过实验评估了StyleGaussian的风格迁移效果、多视角一致性以及运行效率，并与现有方法进行了对比。

#### 实验设置与对比方法

**数据集：** 实验在两个真实场景数据集上进行：Tanks and Temples数据集和Mip-NeRF 360数据集 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=Datasets and Baselines,the weights of the radiance))。这些数据集包含实拍的复杂场景，有较复杂的几何结构和细节，非常具有挑战性 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=Datasets and Baselines,the weights of the radiance))。风格图像方面，作者从WikiArt美术数据库中选取了大量艺术画作为训练用的风格集合 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=,On the))。此外还选用了多种风格的额外图像来测试泛化性，涵盖不同流派和颜色搭配 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=,On the))。

**基线方法：** 作者将StyleGaussian与两种最新的**零样本3D风格迁移**方法进行了对比 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=images dataset,Unlike our))：

- **HyperNet（Chiang et al. 2022）** ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=method with two state,shot methods))：这是基于NeRF的隐式风格迁移方法。它使用一个NeRF MLP来重建场景辐射场，然后训练一个“超网络”根据输入风格图像生成NeRF网络某些权重的偏移，实现辐射场随风格变化 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=method with two state,shot methods))。换言之，不同风格通过调整NeRF网络参数来体现。该方法可以处理任意风格，但存在**渲染慢、风格细节欠缺**的问题 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=images and better content preservation,the reference style images with))。由于仍以NeRF为底层，每帧渲染需要MLP高密度采样计算，很难实时 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=a radiance field for a,shot methods))。并且通过超网络调参数的方式，精细的笔触纹理难以逼真再现 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=images and better content preservation,the reference style images with))。
- **StyleRF（Huang et al. 2022）** ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=prediction branch using a hypernetwork,based style transfer))：该方法使用**TensoRF**（一种用张量分解表示辐射场的NeRF变体）来表示场景。 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=prediction branch using a hypernetwork,based style transfer))首先将VGG图像特征嵌入到重建的TensoRF体元中（和StyleGaussian的嵌入思路类似），然后**在渲染出的2D特征图上**进行AdaIN风格变换，接着用一个2D CNN将风格化特征解码为图像 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=prediction branch using a hypernetwork,based style transfer))。StyleRF的特点是不需要针对新风格训练——它直接应用AdaIN实现任意风格迁移，因而也是零样本的。然而，它采用2D CNN解码，**多视角一致性较差**，有时会出现条纹伪影等 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=to the hypernetwork’s limited ability,the stylized RGB in 3D))。另外，TensoRF渲染虽然比NeRF快一些，但每帧仍需几十到上百毫秒，不够实时 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=LPIPS RMSE LPIPS RMSE Seconds,005))。

作者没有拿那些需要测试时优化的风格迁移方法（每个风格要优化10分钟以上）来比，因为速度上显然不具有可比性 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=field and embeds VGG features,24 %2C  14))。也没有和基于网格或点云的风格化方法比，这是因为根据此前研究，这类方法在重建真实场景上往往有较大误差，无法提供高质量的内容基准 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=methods%2C as their transfer time,24 %2C  14))。

#### 定性结果

作者首先展示了各方法的**视觉效果**对比。结果如论文Fig.5所示，每种方法对若干场景-风格组合的输出被并排比较 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=We show the qualitative results,stylized novel views%2C which are))。主要观察结论如下：

- **StyleGaussian：** 效果最好，风格纹理与参考风格图非常贴近，颜色和画风都“对味” ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=We show the qualitative results,stylized novel views%2C which are))。同时，场景内容细节（物体结构、边缘等）保存得很好 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=We show the qualitative results,stylized novel views%2C which are))。肉眼看去，StyleGaussian的风格化结果既充分体现了艺术风格，又不会让人认不出原来的场景。
- **HyperNet：** 效果较差。它往往**无法正确捕捉风格**，经常只是给场景覆上一层单调的颜色，缺乏应有的纹理变化，画面有种过度平滑、颜色单一的感觉 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=images and better content preservation,the reference style images with))。例如对本应具有笔触的风格，HyperNet可能仅产生颜色上的平均变化，没有明显纹理。这被认为是由于超网络难以用少量参数调整来刻画任意图像风格的丰富细节 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=images and better content preservation,the reference style images with))。总之，HyperNet的结果比较“素”，很多风格特征没能迁移过来。
- **StyleRF：** 比HyperNet有明显改进，能够呈现一定的风格纹理 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=to the hypernetwork’s limited ability,the stylized RGB in 3D))。然而，StyleRF在某些风格上仍然力有不逮，例如论文指出在某些示例（如图中第3、7、8行）StyleRF未能完全表现出参考风格的颜色基调 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=to the hypernetwork’s limited ability,the stylized RGB in 3D))。此外，更严重的是，它在生成的新视角图像中出现了一些不自然的**杂波和条纹状纹理** ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=However%2C it still struggles to,the stylized RGB in 3D))。这些伪影很可能源于其使用2D CNN解码，导致风格在不同视角下不稳定，出现错位叠加，从而在图像中表现为条纹状失真 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=However%2C it still struggles to,the stylized RGB in 3D))。
- **StyleGaussian** 的优势进一步体现：它的风格化结果既没有HyperNet那样丢失风格，也不像StyleRF出现明显伪影 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=unwanted artifacts and striped textures,the stylized RGB in 3D))。无论颜色还是纹理，都与风格参考图高度一致，而且在不同视角下的图像干净且一致。这种**优异的风格迁移质量**很大程度上归功于StyleGaussian的设计——直接在3D解码出风格化颜色，避免了2D处理导致的不一致 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=unwanted artifacts and striped textures,the stylized RGB in 3D))。

简单来说，从肉眼对比，StyleGaussian在**风格相似度**和**内容保真度**两方面都胜过其他方法，能够产生赏心悦目的3D风格化效果 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=We show the qualitative results,stylized novel views%2C which are))。

#### 定量结果

评价3D风格迁移质量是一个难点，目前没有统一的标准指标 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=As pointed out in previous,decode the RGB values%2C maintaining))。作者参考前人做法，从**多视角一致性**和**效率**两个方面进行了量化比较 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=As pointed out in previous,decode the RGB values%2C maintaining))：

- **多视角一致性指标：** 他们采用一种基于光流的方法来度量 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=perform quantitative comparisons on multi,StyleGaussian also outperforms HyperNet%2C since))。具体做法是选取场景的两个视角渲染结果，将一个视角的图像通过计算光流场warp到另一个视角，然后计算这两张图像warp后重叠区域的差异。如果风格是一致的，这两个视角对应区域颜色应相似；反之若不一致，差异会大。采用的度量包括**RMSE**（颜色均方根误差）和**LPIPS**感知距离 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=perform quantitative comparisons on multi,StyleGaussian also outperforms HyperNet%2C since))（后一种衡量感知上的图像差异）。他们针对视角较近（short-range）和视角差异较大（long-range）分别计算了指标 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=Table 1%3A Quantitative results,range Consistency Transfer Time)) ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=Time LPIPS RMSE LPIPS RMSE,005))。结果显示，**StyleGaussian的RMSE和LPIPS远低于StyleRF**，表明其视角一致性要好得多 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=,in the style transfer task))。这归因于StyleGaussian使用3D CNN解码保证了一致性，而StyleRF用2D CNN每视角独立处理导致不一致 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=,in the style transfer task))。相比HyperNet，StyleGaussian的一致性也更好，因为HyperNet的NeRF本身带有视角依赖效果，风格在不同方向会略有变化 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=margin%2C as StyleGaussian uses a,in the style transfer task))。因此，在保持风格连贯性方面，StyleGaussian领先。
- **时间性能：** 论文将**风格迁移耗时**和**渲染耗时**也进行了比较 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=Time LPIPS RMSE LPIPS RMSE,005))。“迁移时间”指给定一个新风格从内容场景生成风格化3D表示所需时间；“渲染时间”指得到该表示后渲染每张视图图像的时间，不包括IO。 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=Time LPIPS RMSE LPIPS RMSE,005))表格结果：StyleGaussian风格迁移时间约0.105秒，**比HyperNet快18倍，比StyleRF快32倍** ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=Furthermore%2C StyleGaussian significantly outperforms others,capability of 3DGS%2C as it))。这种巨大加速源于StyleGaussian对整个3D表示一次性进行风格转换，并在GPU上高效并行计算，相比之下HyperNet/StyleRF需要对NeRF/TensoRF逐像素处理，速度慢很多 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=speed,They need to))。更惊人的是渲染速度：StyleGaussian每帧0.005秒（约200fps），保持了3DGS的实时性能 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=HyperNet ,005))。而HyperNet渲染一帧要32.3秒，StyleRF约10.1秒 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=Time LPIPS RMSE LPIPS RMSE,005))。换算下来，StyleGaussian渲染速度**比HyperNet快约6000倍，比StyleRF快约2000倍** ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=StyleRF intertwine the rendering and,to HyperNet and TensoRF%2C respectively))。这主要因为后两者将风格迁移和渲染耦合在了一起，**每渲染一张新视图都要执行风格网络**，既没有利用内容的重复，又叠加了原本NeRF/TensoRF就偏慢的渲染 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=StyleGaussian maintains the real,to HyperNet and TensoRF%2C respectively))。相反，StyleGaussian将风格应用在3D数据上且一次搞定，新视图只是普通的高斯渲染，几乎零开销 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=applies style transfer to the,Quantitatively))。因此在实际应用中，StyleGaussian能够真正实现交互式速度的3D风格迁移，而其他方法难以胜任。

### 消融实验

作者设计了消融实验来验证各模块设计的有效性，包括**解码器结构**和**内容-风格平衡**两个方面。

#### 解码器结构对比

这部分实验将StyleGaussian的3D卷积解码器与其他两种可能的解码方式进行了对比 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=KNN,the superior capability of the))：

- **逐点MLP解码：** 用一个简单的全连接网络（感知机）直接将每个高斯的风格特征映射到RGB。不考虑任何邻域信息，相当于每个点独立上色。
- **PointNet解码：** 使用PointNet风格的对称函数聚合邻域特征，再输出每个点颜色。PointNet对邻域的处理是将邻域所有点特征经过多层感知机后取**最大值**（或平均值）聚合，然后再映射输出。这保证了对邻域点排列不敏感，但可能丢失部分信息。

实验结果通过图像示例（Fig.6）展示 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=Image%3A Refer to caption ,stylization effect by adjusting))。结论如下：

- PointNet解码**失败**了：产生的风格化结果中，每个高斯的颜色几乎都是一样的，场景看起来像蒙上单色滤镜，没有正确应用局部的风格变化 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=KNN,the superior capability of the))。原因在于PointNet的对称聚合使得邻域不同点云输出同一个特征，导致所有点云获得相同的颜色 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=KNN,the superior capability of the))。这显然不足以表达复杂的风格纹理。
- MLP解码有**一定效果**：它能为不同位置赋予不同颜色，表现出一些风格倾向，但整体来说风格的纹理细节不丰富 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=color variations%2C resulting in each,the superior capability of the))。尤其是那种需要连续笔触或大范围纹理的风格，单点MLP无法捕捉，因为它看不到邻居点，只能基于自身特征决定颜色 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=color variations%2C resulting in each,the superior capability of the))。因此风格往往局限于颜色变化，缺少纹理。
- KNN 3D卷积解码效果**最佳**：能够产生丰富的色彩变化和纹理细节，没有出现所有点同色的问题 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=KNN,the superior capability of the))。尤其对于需要连续纹理（比如画作笔触）的风格，KNN卷积通过邻域融合，成功在点云表面形成连贯的风格花纹。这验证了**作者设计的3D CNN确实在解码阶段比其他方案更有优势** ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=cause a group of points,3D CNN in these aspects))。大的感受野和邻域交互使它既不会丢失局部差异，又能在局部范围内保持连贯性。

#### 内容-风格平衡

风格迁移中常常需要在“保持内容”和“体现风格”之间找到平衡。为此，作者在训练损失中引入了权重系数λ\lambda来调节内容损失和风格损失的比重（见上文训练细节） ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=Content,yields visually most pleasant results))。在消融实验中，他们通过调整λ\lambda观察最终效果 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=Content,yields visually most pleasant results))。

当λ\lambda较高时，内容损失占比大，优化倾向于保留更多原始内容细节，风格的颜色和纹理可能相对弱化；反之λ\lambda低则更强调风格效果，但可能使内容结构扭曲或颜色偏离原场景 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=Content,yields visually most pleasant results))。实验发现，选择一个适中的λ\lambda可以在两者间取得视觉上最令人满意的结果 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=Content,yields visually most pleasant results))。作者在文中提到，通过调节这一系数，可以让生成结果偏向内容或偏向风格。例如，当希望**多保留内容**时，可增大λ\lambda，得到更清晰的场景结构；当希望**更夸张地表现风格**时，可减小λ\lambda，让图像颜色纹理更接近艺术风格。最终实验选择了能平衡二者的λ\lambda值，使得结果在主观视觉上既保留场景识别性又具有浓郁的艺术风格 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=Content,yields visually most pleasant results))。

### 应用

在论文的应用部分，作者展示了StyleGaussian方法的两个有趣拓展：**风格插值**和**解码器初始化迁移**。

#### 风格插值

StyleGaussian支持在推理阶段对**不同风格进行平滑插值** ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=Style interpolation,view consistency))。具体来说，假设我们有两张风格图像 Is1I_{s1} 和 Is2I_{s2}。通过AdaIN，我们可以得到场景高斯的两套风格特征 Gt1G^{t_1} 和 Gt2G^{t_2}（分别对应风格1和风格2）。由于风格特征存在于相同的3D高斯空间，我们可以直接对这两套特征按照某个插值系数 α\alpha 做线性插值：fpinterp=αfpt1+(1−α)fpt2f^{interp}_p = \alpha f^{t1}_p + (1-\alpha) f^{t2}_p。这会得到一套介于两种风格之间的过渡特征 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=Style interpolation,view consistency))。然后使用之前训练好的3D解码器对插值特征解码，即可生成一种**混合风格**的3D场景 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=Style interpolation,view consistency))。

作者在Fig.8展示了连续改变α\alpha的结果 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=Image%3A Refer to caption ,than of the training time))：可以看到风格效果在两端之间平滑过渡。例如，一端是梵高风格、一端是莫奈风格，中间值则呈现两者元素混杂的绘画风格，而且过渡自然，没有明显的断层。这说明StyleGaussian不仅能迁移单一风格，还能在同一场景上无缝融合多种风格。重要的是，插值后的高斯仍然可以实时渲染，保持多视角一致性 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=interpolation is achieved by blending,view consistency))。这对于创意应用非常有用，比如在动画或VR中动态调整场景的艺术风格。

#### 解码器初始化迁移

尽管StyleGaussian实现了任意风格的零样本迁移，但**每个新场景**在使用前仍需要经过前述的预训练过程（特征嵌入和解码器训练）。这个预处理对复杂场景来说可能比较耗时，是包括StyleGaussian在内的所有零样本3D风格迁移方法的共同限制 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=Decoder initialization,than of the training time))。为降低每个新场景的训练成本，作者探索了**解码器参数迁移**的方法 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=Decoder initialization,than of the training time))。

他们发现，不同场景的风格解码任务有一定共性，因此可以将一个场景训练好的3D解码器**作为另一个场景解码器的初始化** ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=consuming pre,than of the training time))。具体做法：对一个新场景，在训练它的解码器时，不从随机初始化开始，而是加载一个已经训练好的旧场景解码器的权重作为起点，然后进行微调 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=consuming pre,than of the training time))。这样一来，新场景的解码器更快收敛。据论文Fig.9显示，用这种初始化的解码器（图(c)）生成的风格化结果，与从零训练的解码器（图(d)）几乎一样，但训练时间却减少到原来的**不到十分之一** ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=Figure 9%3A Decoder initialization,than of the training time)) ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=consuming pre,than of the training time))（作者提到只需原来不到10%的时间）。这对于实际应用很有意义：如果有一系列场景需要风格化，我们可以先在一个代表性场景上训练好解码器，然后在其他场景上快速微调，大大加快处理速度。

### 局限性

尽管StyleGaussian表现出色，作者也诚恳地指出了该方法的两个主要局限：

1. **仅改变颜色，不改动几何：** StyleGaussian目前**只修改3D高斯的颜色**，不改变它们的空间位置或形状 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=Our method has two major,for a 24G memory capacity))。这意味着某些与几何相关的风格（比如夸张的变形、线条勾勒等）无法通过该方法实现。如果原场景的形状与风格不符，StyleGaussian也只能在颜色上近似，无法引入新的几何元素或笔触 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=Our method has two major,for a 24G memory capacity))。作者指出，目前大多数零样本辐射场风格迁移都专注于颜色风格化，保持几何不变。但未来一个有趣的方向是**探索零样本的几何风格迁移**，让辐射场的形状也能随风格改变 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=some style patterns cannot be,which could save considerable memory))。这将进一步提高风格迁移的表现力。
2. **内存和计算随高斯数量增加：** 虽然StyleGaussian的风格迁移模块参数量与高斯数无关（例如解码器的卷积核大小固定），但实际运行时，**需要对每个高斯存储和计算高维特征**，所以内存和计算开销仍然**随场景高斯数量线性增加** ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=fields presents an interesting direction,which could save considerable memory))。在实验中，为适应24GB显存，他们限制了每个场景的高斯数量（具体数值论文中有给出，如几个百万） ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=in StyleGaussian’s style transfer module,shot style transfer without))。这在一定程度上可能影响极大场景或更高细节的表现。作者建议未来的研究可以考虑在不显式存储每个高斯高维特征的情况下实现风格迁移 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=Gaussians%2C both memory and computation,which could save considerable memory))。例如，可否通过某种隐式网络来生成特征，或者对高斯进行分层压缩表示等。如果能减少特征存储，将显著节省内存并可能加速计算。

### 结论

文章最后总结了StyleGaussian的方法和意义。作者提出了一种**全新的3D风格迁移方法**，能够将任意图像的艺术风格**即时**地迁移到重建的3D高斯场景中 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=In this paper%2C we introduce,based 3D CNN as the))。StyleGaussian在不损失3D高斯铺撒原有的实时渲染和多视角一致性的前提下，实现了风格编辑 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=In this paper%2C we introduce,based 3D CNN as the))。

方法包含三个步骤：特征嵌入、风格变换、特征解码，分别完成嵌入VGG特征、风格化特征和解码RGB颜色 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=Gaussians of a scene,to be integrated into various))。为克服高维特征渲染的困难，作者引入了高效特征渲染策略，成功将高维VGG特征嵌入3D表示 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=Splatting,of artistic styles into 3D))。同时开发了KNN邻域的3D卷积解码器，具备大的感受野且确保了严格的多视角一致性 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=Splatting,of artistic styles into 3D))。

实验结果验证了StyleGaussian的优越性：即时的风格迁移（无需针对新风格的优化），远超前人的渲染速度，以及出色的风格质量和一致性。这样的性能使得StyleGaussian具有很高的应用潜力 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=decodes the features to RGB%2C,artistic styles into 3D environments))。作者展望，该方法可以集成到增强现实（AR）、虚拟现实（VR）、电子游戏和电影制作等应用中，将艺术风格无缝融入3D环境 ([[2403.07807\] StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://ar5iv.org/abs/2403.07807#:~:text=rendering strategy to address the,artistic styles into 3D environments))。想象一下，游戏开发者可以即时地改变场景的画风，或者用户在VR中一键切换所见世界的艺术风格——这些都有望通过StyleGaussian得以实现。

总之，StyleGaussian为3D内容的风格编辑开启了新的可能。它结合了3D高斯铺撒的高速渲染和神经风格迁移的强大表现力，**在3D场景中实现了前所未有的即时艺术风格转换**，为相关领域的研究和应用奠定了坚实基础。 ([StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting](https://kunhao-liu.github.io/StyleGaussian/#:~:text=render the high,view consistency))