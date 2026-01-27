输入：一个视频+目标轨迹

输出：对应目标轨迹的视频

## 方法

1. 构建数据集：使用UE5渲染生成40个环境，13.6K个动态场景，每个场景10个相机（136K个视频，每个视频带外参轨迹），122K条轨迹
2. 从数据集取：source视频Vs，target视频Vt，target轨迹CAMt，视频的文本caption（captioner生成）

3. 修改模型：
	- 基于预训练T2V模型（具体是哪个模型？Wan2.1 T2V），给他加一个相机编码器输出通道数等于spatial attention的输出通道数，将相机embedding与spatial attention的输出相加
	- 预训练模型的latent输入是zt，这里加一条路径：输入source视频的的 latent，然后patichify，然后在f维度上拼接输入diffusion
	- 仅训练：相机编码器，3Dattention， Projector（这个projector是啥架构？干嘛用的？哪来的？你可以把 projector 当成“把 token 做一个简单线性变换的小头”）

4. 监督训练：拿什么来监督？预测噪声还是用target视频？

5. 训练采用三种模式：
	- 60%：V2V生成：source视频的latent正常输入
	- 20%：T2V生成：source视频的latent随机初始化
	- 20%：I2V生成：source视频的latent仅用第一帧，后面几帧随机初始化









结合常见 DiT 结构和开源代码结构（有 `wan2p1_dit.py` 这类文件），比较合理的推断是：

> **Projector 就是一层（或几层）线性 / 卷积投影层**，用来把「拼接过的 token」或某些中间特征，重新投影到 backbone DiT 需要的特定通道空间里，以避免破坏预训练分布。