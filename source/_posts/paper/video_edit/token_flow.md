---
title:  "Text2Video-Zero:  Text-to-Image Diffusion Models are Zero-Shot Video Generators论文理解"

date:  2025-2-14 13:28:00

tags:  [视频编辑,论文]

categories:  [论文,视频]

comment:  false

toc:  true




---

#

<!--more-->

- ICLR 2024
- [论文地址](https://arxiv.org/abs/2307.10373)
- [项目地址](https://github.com/omerbt/TokenFlow)

# 0. Abstract

- 生成式人工智能革命最近扩展到了视频领域。然而，就视觉质量和用户对生成内容的控制而言，当前最先进的视频模型仍然落后于图像模型。在这项工作中，我们提出了一个框架，**利用文本到图像扩散模型的力量来完成文本驱动的视频编辑任务**。具体来说，给定源视频和目标文本提示，我们的方法生成坚持目标文本的高质量视频，同时保留输入视频的空间布局和运动。我们的方法基于一个**关键的观察结果**，即**编辑视频中的一致性可以通过在扩散特征空间中实施一致性来获得**。我们通过**基于帧间对应关系显式传播扩散特征来实现这一点**，这在模型中很容易获得。因此，我们的框架**不需要任何训练或微调**，可以与任何现成的文本到图像编辑方法结合使用。我们在各种真实视频上展示了最先进的编辑结果。

# 1. Introduction

- 我们的目标是生成符合输入文本提示所表达的目标编辑的高质量视频，同时保留原始视频的空间布局和运动。

- 利用图像扩散模型进行视频编辑的主要挑战是**确保编辑的内容在所有视频帧中保持一致**——理想情况下，3D世界中的每个物理点都会随着时间的推移进行连贯的修改。基于图像扩散模型的现有和并发视频编辑方法已经证明，**通过将自我注意模块扩展到包括多个帧，可以实现跨编辑帧的全局外观一致性**（TAV, Text2video-zero, Pix2video, Fatezero）

- 然而，这种方法不足以实现期望的时间一致性水平，**因为视频中的运动仅通过注意力模块隐式地保留**。因此，专业或半专业用户经常求助于复杂的视频编辑管道，这需要额外的手动工作。
- 在这项工作中，我们提出了一个框架，通过**显式地加强编辑上的原始帧间对应**来解决这一挑战。直观地，**自然视频包含跨帧的冗余信息**，例如，描绘**相似的外观和共享的视觉元素**。我们的关键观察是，**扩散模型中 视频的内部表示 表现出类似的属性**。也就是说，**RGB空间和扩散特征空间中的帧的冗余水平和时间一致性是紧密相关的**。基于这一观察，**我们方法的支柱是通过确保 *结果视频的特征* 跨帧一致来实现一致的编辑**。具体来说，我们**强制编辑的特征传达与原始视频特征相同的帧间对应和冗余**。为此，我们利用模型容易获得的**原始帧间特征对应关系**。这导致了一种有效的方法，该方法可以**直接基于原始视频动态传播编edited diffusion特征** 。这种方法允许我们利用最先进的**图像扩散模型**的生成先验，而无需额外的训练或微调，并且可以与现成的基于扩散的图像编辑方法结合工作（例如，SDEdit；prompt2prompt；controlnet；plug and play）。总而言之，我们做出了以下主要贡献：
  - 一种被称为TokenFlow的技术，它强制跨帧扩散特征的语义对应，允许显著增加由文本到图像扩散模型生成的视频的时间一致性。
  - 研究视频中扩散特征特性的新颖实证分析。
  - 对不同视频进行最先进的编辑结果，描绘复杂的运动。

# 2. Related Work

## 2.1 文生图&文生视频

利用预先训练的图像扩散模型进行视频合成任务，无需额外训练

- Scenescape
- TAV
- Shape-aware textdriven layered video editing
- Fatezero

## 2.2 一致的视频风格化

- 常见方法包括在逐帧的基础上应用图像编辑技术（例如，风格转移），然后进行后处理阶段，以解决编辑视频中的时间不一致（Learning blind video temporal consistency；Blind video temporal consistency via deep video prior；Blind video deflickering by neural filtering with a flawed atlas）。
- 尽管这些方法有效地减少了高频时间闪烁，但**它们并没有被设计成处理在内容上表现出实质性变化的帧**（they are not designed to handle frames that exhibit substantial variations in content），这在应用基于文本的图像编辑技术时经常发生（Fatezero）
- Layered neural atlases for consistent video editing. 提出将视频分解成一组2D图谱，每个图谱在整个视频中提供背景或前景对象的统一表示。应用于2D地图集的编辑会自动映射回视频，从而以最小的努力实现时间一致性。Text2live；Shape-aware text-driven layered video editing demo利用这种表示来执行文本驱动的编辑。然而，图谱表示仅限于具有简单运动的视频，并且需要长时间的训练，限制了该技术和基于该技术的方法的适用性。
- 我们的工作也与经典工作有关，这些工作证明了**自然视频中的小块在帧之间广泛重复**（Space-time super-resolution from a single video；Video epitomes），因此一致的编辑可以通过 *编辑关键帧的子集并通过使用手工制作的特征和光流建立补丁对应关系在整个视频中传播编辑*  （Ruder等人，2016；Stylizing video by example）或  *通过训练基于补丁的GAN* （Texler等人，2020年）
- 然而，这种传播方法难以处理**具有照明变化或具有复杂动态**的视频。重要的是，它们依赖于用户提供的关键帧的一致编辑，这仍然是尚未自动化的劳动密集型任务。rerender a video将关键帧编辑与Stylizing video by example的传播方法相结合。他们使用文本到图像扩散模型编辑关键帧，同时对编辑的关键帧实施光流约束。**然而，由于远距离帧之间的光流估计不可靠，他们的方法无法一致地编辑相距很远的关键帧**（如我们的补充材料-SM中所见），因此无法一致地编辑大多数视频。

- 我们的工作与这种受益于自然视频中时间冗余的方法有着相似的动机。我们表明，这种冗余也存在于文本到图像扩散模型的特征空间中，并利用这一特性来实现一致性。

## 2.3 通过扩散特征操纵的受控生成

- 最近，大量工作展示了文本到图像扩散模型如何通过对扩散网络的中间特征表示执行简单的操作来轻松适应各种编辑和生成任务（Attend-and-excite: Attention-based semantic guidance for text-to-image diffusion models；Improving sample quality of diffusion models using self-attention guidance；Directed diffusion；Plug-and-play diffusion features for text-driven image-to-image translation；Prompt-to-prompt；Localizing object-level shape variations with text-to-image diffusion models；Masactrl: Tuningfree mutual self-attention control for consistent image synthesis and editing）。
- Diffusion hyperfeatures: Searching through time and space for semantic correspondence;   A tale of two features: Stable diffusion complements dino for zero-shot semantic correspondence.演示了使用扩散特征对应的语义外观交换。Prompt-to-prompt观察到，通过操纵交叉注意层，可以控制图像的空间布局与文本中每个单词之间的关系。Plug-and-Play Diffusion分析了空间特征和self attention map