



|               | gram       | artfid    | lpips-1      | mse-1        | lpips-10     | mse-10       |
| ------------- | ---------- | --------- | ------------ | ------------ | ------------ | ------------ |
| Ours          | 0.91e8     | **10.37** | **0.054662** | **0.021264** | **0.141878** | **0.067088** |
| SGSST         | 1.10e8     | 11.60     | 0.061608     | 0.028316     | 0.191510     | 0.088421     |
| stylegaussian | 2.09e8     | 16.71     | **0.053229** | 0.023736     | 0.165190     | 0.078296     |
| StylizedGS    | 3.17e8     | 14.91     | 0.069652     | 0.027369     | 0.242991     | 0.083439     |
| CCPL          | 0.49e8     | 15.41     | 0.115843     | 0.057476     | 0.210965     | 0.098498     |
| CAP-VSTNet    | 0.68e8     | 14.78     | 0.073297     | 0.030320     | 0.174318     | 0.075855     |
| ReReVST       | 1.95e8     | 13.86     | 0.078171     | 0.024805     | 0.180283     | **0.061089** |
| AdaIN         | **0.43e8** | 12.18     | 0.151387     | 0.070549     | 0.267297     | 0.114765     |
| aespa         | 1.36e8     | 14.74     | 0.176603     | 0.068057     | 0.257548     | 0.099323     |
| samam-4       | 1.02e8     | 15.61     | 0.140929     | 0.073707     | 0.244332     | 0.117091     |
| Adaattn       | 0.72e8     | 12.61     | 0.153529     | 0.065502     | 0.236853     | 0.101632     |
| artflow       | 0.81e8     | 14.28     | 0.176154     | 0.069034     | 0.274072     | 0.107173     |
| StyA2k        | 1.19e8     | 15.14     | 0.086351     | 0.040145     | 0.174162     | 0.087453     |

AdaIN             &\textbf{0.43e8}      &12.18     &0.151    &0.071     &0.267     &0.115  \\
AesPA-Net      &1.36e8      &14.74     &0.177    &0.068     &0.258     &0.099  \\
SaMam           &1.02e8      &15.61     &0.141    &0.074     &0.244     &0.117  \\
ReReVST         &1.95e8      &13.86     &0.078    &0.025     &0.180     &\textbf{0.061}  \\
CCPL              &0.49e8      &15.41     &0.116    &0.057     &0.211     &0.098  \\
CAP-VSTNet  &0.68e8      &14.78     &0.073    &0.030     &0.174     &0.076  \\


StyleGaussian & 2.09e8 & 16.71 & \textbf{0.053} & 0.024 & 0.165 & 0.078 \\
SGSST         & 1.10e8 & 11.60 & 0.062          & 0.028 & 0.192 & 0.089 \\
StylizedGS    & 3.17e8 & 14.91 & 0.070          & 0.027 & 0.243 & 0.083 \\





|               | lpips-1      | mse-1        | lpips-10     | mse-10       | gram       | artfid    |
| ------------- | ------------ | ------------ | ------------ | ------------ | ---------- | --------- |
| Ours          | **0.054662** | **0.021264** | **0.141878** | **0.067088** | **0.91e8** | **10.37** |
| SGSST         | 0.061608     | 0.028316     | 0.191510     | 0.088421     | 1.10e8     | 11.60     |
| stylegaussian | **0.053229** | 0.023736     | 0.165190     | 0.078296     | 2.09e8     | 16.71     |
| StylizedGS    | 0.069652     | 0.027369     | 0.242991     | 0.083439     | 3.17e8     | 14.91     |

|            | lpips-1      | mse-1        | lpips-10     | mse-10       | gram   | artfid    |
| ---------- | ------------ | ------------ | ------------ | ------------ | ------ | --------- |
| Ours       | **0.054662** | **0.021264** | **0.141878** | **0.067088** | 0.91e8 | **10.37** |
| CCPL       | 0.115843     | 0.057476     | 0.210965     | 0.098498     | 0.49e8 | 15.41     |
| CAP-VSTNet | 0.073297     | 0.030320     | 0.174318     | 0.075855     | 0.68e8 | 14.78     |
| ReReVST    | 0.078171     | 0.024805     | 0.180283     | **0.061089** | 1.95e8 | 13.86     |

|         | lpips-1      | mse-1        | lpips-10     | mse-10       | gram       | artfid    |
| ------- | ------------ | ------------ | ------------ | ------------ | ---------- | --------- |
| Ours    | **0.054662** | **0.021264** | **0.141878** | **0.067088** | 0.91e8     | **10.37** |
| AdaIN   | 0.151387     | 0.070549     | 0.267297     | 0.114765     | **0.43e8** | 12.18     |
| aespa   | 0.176603     | 0.068057     | 0.257548     | 0.099323     | 1.36e8     | 14.74     |
| samam-4 | 0.140929     | 0.073707     | 0.244332     | 0.117091     | 1.02e8     | 15.61     |

|       | lpips-1  | mse-1    | lpips-10 | mse-10   | gram      | artfid       | CSD          | CLIP_C       | CLIP_S          | SSIM      | LPIPS       | GRAM_5     | GRAM_4     |
| ----- | -------- | -------- | -------- | -------- | --------- | ------------ | ------------ | ------------ | --------------- | --------- | ----------- | ---------- | ---------- |
| Ours  | 0.054662 | 0.021264 | 0.141878 | 0.067088 | 0.91e8    | **10.37**    | *0.529480*   | *64.121224*  | *68.889236*     | 0.593846  | 0.405123    | 0.247090   | 0.245490   |
| color | 0.05403  | 0.021159 | 0.14025  | 0.066115 | 85483774  | 10.5415      | 0.511843     | 64.014322    | 68.070789       | 0.586283  | 0.41235467  | 0.35077196 | 0.349220   |
| edge  | 0.054275 | 0.021852 | 0.140332 | 0.066725 | 89520081  | *10.38403*   | **0.583589** | 60.97860243  | **70.43637152** | 0.50904   | 0.459120106 | 0.21972421 | 0.21812059 |
| pam   | 0.052783 | 0.021416 | 0.142545 | 0.068634 | 122569815 | 10.818285942 | 0.52130497   | **64.32773** | 66.550130208    | 0.6036312 | 0.40151204  |            |            |
|       |          |          |          |          |           | x            | x            | x            |                 | x         |             |            |            |

% w/o latent optimisation
% & 10.8183 & 0.6655 & \textbf{0.6433}  \\
% w/o edge regularisation
% & \underline{10.3840} & \textbf{0.7044} & 0.6098 \\

Table~\ref{tab:ablation} reports the quantitative ablation results. In addition to ArtFID, which reflects the overall stylisation quality, we further include two CLIP-based metrics to evaluate semantic style and content preservation. Specifically, CLIP-S measures the image-level similarity between the stylised rendering and the reference style image, while CLIP-C measures the similarity between the stylised rendering and the corresponding generated content target. Higher CLIP-S and CLIP-C indicate better style alignment and content preservation, respectively. As shown in Table~\ref{tab:ablation}, the full model achieves the best ArtFID and second-best CLIP-S and CLIP-C scores, indicating a favourable balance between stylisation quality, style alignment, and content preservation. Removing the latent-space optimisation leads to a slightly higher CLIP-C, but its lower CLIP-S and worse ArtFID suggest weaker stylisation and less stable appearance optimisation. Removing the edge regularisation yields the highest CLIP-S, but causes a clear drop in CLIP-C, indicating that stronger style transfer is obtained at the expense of content preservation. Removing the chroma regularisation also degrades all three metrics compared with the full model, confirming its contribution to stable and perceptually favourable stylisation.

对于定量比较，我们Following the quantitative evaluation protocol described above, we further use CLIP-S and CLIP-C to measure style alignment and content preservation in the ablation study. As shown in Table~\ref{tab:ablation}, the proposed full model achieves the best ArtFID and the second-best CLIP-S and CLIP-C, showing the most balanced overall performance. Direct RGB optimisation 表现出最差的artfid，表明我们的ed latent-space optimis有利于提升整体质量. Removing edge regularisation减少了clip-c分数, suggesting that 我们的. Removing chroma regularisation also degrades all three metrics, confirming its contribution to stable stylisation quality.

对于定量比较，我们Following the quantitative evaluation protocol described above, we use ArtFID to assess overall stylisation quality, and CLIP-based similarities~\citep{radford2021learning} to measure style alignment (CLIP-S) and content preservation (CLIP-C). As shown in Table~\ref{tab:ablation}, the proposed full model achieves the best ArtFID and competitive CLIP-S/CLIP-C scores, indicating the best overall trade-off. Direct RGB optimisation slightly improves CLIP-C but weakens stylisation quality. Removing edge regularisation increases CLIP-S at the cost of content preservation, while removing chroma regularisation degrades all three metrics.

对于定量比较，我们Following the quantitative evaluation protocol described above, we further use CLIP-based similarities~\citep{radford2021learning} to measure style alignment (CLIP-S) and content preservation (CLIP-C). As shown in Table~\ref{tab:ablation}, the full model achieves the best ArtFID and the second-best CLIP-S and CLIP-C, indicating the most balanced overall performance. Direct RGB optimisation gives the worst ArtFID, showing that latent-space optimisation improves overall stylisation quality. Removing edge regularisation clearly reduces CLIP-C, confirming its role in content preservation. Removing chroma regularisation degrades 整体质量与风格分数, demonstrating its contribution to 保持风格一致性.

For quantitative comparison, we follow the evaluation protocol described above and further use CLIP-based similarities~\citep{radford2021learning} to measure style alignment (CLIP-S) and content preservation (CLIP-C). As shown in Table~\ref{tab:ablation}, the full model achieves the best ArtFID and the second-best CLIP-S and CLIP-C, indicating the most balanced overall performance. Direct RGB optimisation gives the worst ArtFID, showing that the proposed latent-space optimisation improves overall stylisation quality. Removing edge regularisation clearly reduces CLIP-C, confirming its role in content preservation. Removing chroma regularisation degrades both overall quality and style alignment, demonstrating its contribution to maintaining style consistency.

| pam     | 2   | 4   | 5   | 11  | 14  | 17  | 20  | 21  | 22  | 26  |
| ------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| counter |     |     |     | x   |     | x   |     |     |     |     |
| family  |     |     |     | x   |     |     |     |     | x   |     |
| garden  |     |     |     |     |     |     |     | x   |     |     |
| horns   |     |     |     | x   |     | x   |     |     |     |     |
| horse   |     |     |     | x   |     |     |     |     | x   |     |
| room    |     |     |     | x   |     | x   |     |     | x   |     |
| train   |     |     |     | x   |     |     |     |     |     |     |
| trex    |     |     | x   | x   |     | x   |     |     |     |     |
| truck   |     |     |     | x   |     | x   |     |     | -   | -   |



| Method        | LC       | LS       | SSIM     | LPIPS    | GRAM_5   | GRAM_4   | CSD          | CLIP_C    | CLIP_S        |
| ------------- | -------- | -------- | -------- | -------- | -------- | -------- | ------------ | --------- | ------------- |
| Ours          | 2.321410 | 1.677991 | 0.703784 | 0.405123 | 0.247090 | 0.245490 | **0.529480** | 64.121224 | **68.889236** |
| SGSST         | 2.540671 | 1.844484 | 0.666168 | 0.450634 | 0.217819 | 0.216490 | 0.462369     | 63.309505 | 66.464106     |
| stylegaussian | 2.319144 | 2.458223 | 0.375919 | 0.410721 | 0.358070 | 0.356001 | 0.204159     | 70.268142 | 60.255642     |
| StylizedGS    | 3.075410 | 5.608161 | 0.529040 | 0.491995 | 0.802432 | 0.800399 | 0.414973     | 60.861589 | 66.882205     |
| CCPL          | 1.605728 | 1.775947 | 0.817793 | 0.373318 | 0.145901 | 0.143903 | 0.236315     | 73.457769 | 55.079644     |
| CCPL_my       | 1.772443 | 1.714392 | 0.749240 | 0.378017 | 0.126413 | 0.124383 | 0.231559     | 74.184696 | 55.220575     |
| CAP-VSTNet    | 1.253702 | 1.715602 | 0.880694 | 0.315467 | 0.126465 | 0.124480 | 0.203683     | 81.517535 | 54.354644     |
| CAP-VST_my    | 1.247055 | 1.634901 | 0.872284 | 0.302290 | 0.098159 | 0.096245 | 0.217295     | 81.128559 | 54.930122     |
| ReReVST       | 1.428740 | 1.409152 | 0.857290 | 0.335025 | 0.201270 | 0.199733 | 0.280411     | 78.502431 | 57.965668     |
| AdaIN         | 1.941202 | 1.758184 | 0.707018 | 0.481242 | 0.183000 | 0.181113 | 0.288712     | 71.994444 | 58.805425     |
| aespa         | 1.676869 | 1.834370 | 0.835332 | 0.345471 | 0.217814 | 0.215947 | 0.365601     | 75.568056 | 59.892969     |
| samam4        | 1.551844 | 3.397963 | 0.607947 | 0.471751 | 0.702786 | 0.699250 | 0.300690     | 75.505642 | 56.720356     |

| Method | LC        | LS       | SSIM      | LPIPS       | GRAM_5     | GRAM_4       | CSD          | CLIP_C      | CLIP_S        |
| ------ | --------- | -------- | --------- | ----------- | ---------- | ------------ | ------------ | ----------- | ------------- |
| Ours   | 2.321410  | 1.677991 | 0.703784  | 0.405123    | 0.247090   | 0.245490     | **0.529480** | 64.121224   | **68.889236** |
| color  | 2.3298016 | 1.86492  | 0.6642861 | 0.4145111   | 0.3475524  | 0.3459804276 | 0.51233      | 63.906474   | 67.8542115    |
| edge   |           |          | 0.6374761 | 0.459120106 | 0.21972421 | 0.21812059   | 0.583589     | 60.97860243 | 70.43637152   |
| pam    |           |          |           |             |            |              |              |             |               |
![[Pasted image 20260507224250.png]]

Following recent 3D scene stylisation studies, we evaluate both stylisation quality and cross-view texture consistency. To better reveal the consistency behaviour of different stylisation strategies in the single-view generative 3DGS setting, we consider three comparison settings in the quantitative evaluation: generated multi-view renderings followed by image stylisation, generated multi-view renderings followed by video stylisation, and direct 3D scene stylisation. Specifically, the image-based baselines include AdaIN, AesPA-Net, and SaMam; the video-based baselines include ReReVST, CCPL, and CAP-VSTNet; and the 3D baselines include StyleGaussian, SGSST, and StylizedGS. Following the rendering setup of \cite{bahmani2025lyra}, each scene-style example is rendered from 6 trajectories with 31 frames per trajectory, and we randomly sample 8 frames from each example to compute the style-related metrics, while using all 186 rendered frames for the consistency evaluation. Style fidelity is assessed by Gram loss and ArtFID. Specifically, Gram loss measures the discrepancy between the multi-level style statistics of the stylised renderings and the reference style image, while ArtFID evaluates the overall stylisation quality from a distributional perspective.


We consider three comparison settings in the quantitative evaluation:  (1) generated multi-view renderings followed by image stylisation,  (2) generated multi-view renderings followed by video stylisation, and  (3) direct 3D scene stylisation. Specifically, the image-based baselines include AdaIN, AesPA-Net, and SaMam(说明通过$^\dagger$标注) ; the video-based baselines include ReReVST, CCPL, and CAP-VSTNet （说明通过$^\ddagger$标注）; and the 3D baselines include StyleGaussian, SGSST, and StylizedGS(说明通过* 标注). Quantitative results are reported in Table~\ref{tab:quantitative}. The image stylisation methods, including AdaIN, AesPA-Net, and SaMam, can produce plausible stylisation on individual rendered frames, but exhibit relatively weak temporal consistency due to the lack of explicit inter-frame constraints. The video stylisation methods, such as ReReVST, CCPL, and CAP-VSTNet, improve continuity by enforcing temporal coherence on rendered frame sequences, and therefore outperform image-based methods on consistency. Notably, their results are particularly competitive at the larger interval $i{=}10$, showing that 2D temporal regularisation remains effective for longer-gap comparisons on rendered videos. However, their temporal consistency is still generally inferior to ours, since their consistency is imposed after rendering rather than directly at the scene representation level. Among the 3D stylisation baselines, our method achieves the best Gram loss and ArtFID scores while maintaining strong cross-view consistency, demonstrating the best overall balance between stylisation quality and consistency. Notably, unlike reconstruction-based pipelines that rely on COLMAP-style initialisation and multi-view correspondence recovery, our method does not require such preprocessing, thereby avoiding an additional source of reconstruction error that may otherwise affect the final stylisation and view consistency.









| 条目                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | 完成  |     |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --- | --- |
| 题目修改                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | x   | ok  |
| 写清楚，你可以有两种办法。 先video生成，multiview 生成multiview生成之后，直接styletransfer会有不连续的问题；                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |     | ok  |
| g_init                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | x   | ok  |
| 增加img/video方法对比（一个unify的统一渲染，这个比video求consistency更好。）                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | x   | ok  |
| 实验设置写清楚渲染用了多少个视角，metric是啥，怎么个比较法。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |     | ok  |
| 加一个比较lpips heat map                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |     |     |
| multi view的3dgs方法要用colmap做初始化，如果3d点找的不准或者coorbondance 找的不准，我们不需要coorbondance matching或者bound adjustment这个前处理的过程。减少了这个error                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |     | ok  |
| 定性比较跟sgsst要有一个放大图，说明你的两个loss有用                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | x   | ok  |
| method里面的例子换一个图                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | x   | -   |
| G0和Gzp这里想办法用一个立体的表示（就是一张图还是投影成他那个图，然后把g0那个还是画个3D的形式表示出来）现在这个看不出来哪个是为了redner真实图的，哪个是render stylize图的。<br>本身我们告诉他我们之所以要fine tune这个东西就是为了去render这个stylize图嘛。是不是我们这边放个示意会比较好点呢？<br>我知道这个没有指向，但是你要让别人知道这个g0他代表啥，你这个后面升级的gp他是代表这个高斯splatting是render啥的，宁可你那个gzp的画大一点，然后上面用两个小回路，你那个phi和phi-1不需要那么大，用一个回指的小箭头就行。放到上面和下面就行。因为你这篇文章的主要贡献在这，你得把这个地方强化出来。下面那个loss把线捋直了，把他变成比较直观的loss往下放，然后上面那部分得把他放大了，主要贡献得在上面那部分显示出来。下面求loss可以小点，drgb下面生成出来的content跟style画的一样，对齐了（到一行）都行。然后所有箭头都拉直，比如content拐到上面到edge那个，那个edge图直接变成一个乘过来的，一个小一点的图，vgg也可以小。三个loss模块画小放到block里面，形成一个loss module，然后上面尽量画大一点。result后面箭头不要往右走，直接下来就行。把上面的整个生成过程，还有gaussian splatting那些模块尽量用大一点的表述，看的很直观的，我们的新意在哪。然后lossblock放到左下  3/4或  4/5. 绿的在右或者底下拉平。图要对称。 | x   | ok  |
|                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |     |     |




\subsection{Implementation Details}
For comparative evaluation,  we conduct experiments on scenes  selected from the Mip-NeRF 360 \citep{barron2022mip}, Tanks\&Temples \citep{knapitsch2017tanks}, and LLFF \citep{mildenhall2019local} datasets. For each scene, we select one image as the content input and use Lyra \citep{bahmani2025lyra} to generate the initial 3DGS scene. Unless otherwise specified, all experiments are conducted at a rendering resolution of 704 $\times$ 1280. We optimize each scene-style pair for 20,000 iterations using the Adam optimizer with a learning rate of 0.0025. （在这一段恰当的地方说明我们训练follow  \cite{bahmani2025lyra} 的训练和配置，采用6条轨迹进行训练，每条训练轨迹采样121帧，每条测试轨迹采样31帧，计算风格迁移损失时取1帧进行反向传播。）

We adopt two scales for the multiscale stylization objective, namely a half-resolution scale and the original rendering scale, and assign them equal weights. In practice,  we optimize only the coarsest scale during the first 10,000 iterations and then jointly optimize both scales during the remaining 10,000 iterations. The weight of the content term is set to 5.0. For the additional regularization terms, the weights of the edge-consistency loss and chroma-consistency loss are set to 0.05 and 0.03, respectively, and the trimming ratio used for style-side chroma statistics is set to 0.05. For fair comparison, all methods are evaluated under matched output resolutions and camera trajectories. 

In implementation, the gradients of the main multiscale style term and the two auxiliary regularization terms are combined by a gradient-balancing strategy, and PCGrad~\citep{yu2020gradient} is used to alleviate gradient conflicts between different objectives. Our implementation is based on PyTorch, and all experiments are conducted on a single NVIDIA L40S GPU. 


\subsection{Implementation Details}
We conduct experiments on scenes selected from the Mip-NeRF 360 \citep{barron2022mip}, Tanks\&Temples \citep{knapitsch2017tanks}, and LLFF \citep{mildenhall2019local} datasets. For each scene, we select a single image as the content input and use Lyra \citep{bahmani2025lyra} to generate the initial 3DGS scene. Unless otherwise specified, all experiments are performed at a rendering resolution of 704 $\times$ 1280. Following the training protocol and camera setup of \cite{bahmani2025lyra}, the generative initialization is based on six camera trajectories, where each training trajectory contains 121 frames and each test trajectory contains 31 frames. During the subsequent stylization optimization, we sample one target frame at each iteration to compute the rendering-domain stylization losses and perform backpropagation. We optimize each scene-style pair for 20,000 iterations using the Adam optimizer with a learning rate of 0.0025.

We adopt two scales for the multiscale stylization objective, namely a half-resolution scale and the original rendering scale, and assign them equal weights. In practice, we optimize only the coarsest scale during the first 10,000 iterations and then jointly optimize both scales during the remaining 10,000 iterations. The weight of the content term is set to 5.0. For the additional regularization terms, the weights of the edge-consistency loss and chroma-consistency loss are set to 0.05 and 0.03, respectively, and the trimming ratio used for style-side chroma statistics is set to 0.05. For fair comparison, all methods are evaluated under matched output resolutions and camera trajectories.

In implementation, the gradients of the main multiscale style term and the two auxiliary regularization terms are combined by a gradient-balancing strategy, and PCGrad~\citep{yu2020gradient} is used to alleviate gradient conflicts between different objectives. Our implementation is based on PyTorch, and all experiments are conducted on a single NVIDIA L40S GPU.

我觉得应该在这一句的基础上继续修改，说我们取得了较好的一致性，由于我们的方法无需引入colmap等初始化重建，which可能引入error，由好的一致性引出这么一个解释。
Among the 3D stylization baselines, our method achieves the best Gram loss and ArtFID scores while maintaining strong cross-view consistency, demonstrating the best overall balance between stylization quality and consistency.



Fortunately, recent advances in camera-controllable video diffusion~\citep{voleti2024sv3d,xu2024cavia} and feed-forward generative 3D content creation~\citep{bahmani2025lyra,tang2024lgm} provide a new path toward this goal. In particular, single-image generative 3DGS models make it possible to directly obtain a view-consistent 3D Gaussian scene from only one image~\citep{bahmani2025lyra,roh2025catsplat}, thereby opening a new setting for 3D style transfer with substantially lower content acquisition cost. However, directly extending existing optimization-based 3DGS stylization methods to this setting remains non-trivial. On the one hand, in our setting Gaussian appearance is explicitly represented by bounded RGB values, and directly optimizing these exported colors over long iterations can induce deviations from the valid bounded RGB domain, leading to noticeable color drift, such as spurious color patches and local color inconsistency, as illustrated in \autoref{introduce_problem}(a). On the other hand, the single-image generative setting makes it more difficult to preserve local structural details and maintain chromatic stability; as shown in \autoref{introduce_problem}(b), the latter issue becomes particularly noticeable under style references with weak or highly restricted chromatic cues, where undesired blue/orange tints may appear in visually neutral regions.

Fortunately, recent advances in camera-controllable video diffusion~\citep{voleti2024sv3d,xu2024cavia} and feed-forward generative 3D content creation~\citep{bahmani2025lyra,tang2024lgm} provide a new path toward this goal. In particular, single-image generative 3DGS models make it possible to directly obtain a view-consistent 3D Gaussian scene from only one image~\citep{bahmani2025lyra,roh2025catsplat}, thereby opening a new setting for 3D style transfer with substantially lower content acquisition cost. A straightforward alternative in this setting is to first generate a multi-view image sequence and then directly apply image or video style transfer to the generated views. However, such a strategy performs stylization only in image space after generation and does not explicitly optimize a unified 3D representation, making it less suitable for maintaining scene-level coherence across viewpoints. We therefore instead stylize the generated 3DGS scene itself under differentiable rendering. However, directly extending existing optimization-based 3DGS stylization methods to this setting remains non-trivial. On the one hand, in our setting Gaussian appearance is explicitly represented by bounded RGB values, and directly optimizing these exported colors over long iterations can induce deviations from the valid bounded RGB domain, leading to noticeable color drift, such as spurious color patches and local color inconsistency, as illustrated in \autoref{introduce_problem}(a). On the other hand, the single-image generative setting makes it more difficult to preserve local structural details and maintain chromatic stability; as shown in \autoref{introduce_problem}(b), the latter issue becomes particularly noticeable under style references with weak or highly restricted chromatic cues, where undesired blue/orange tints may appear in visually neutral regions.

counter 大
family 小了
horns小




figure 4的图2换一张
adain换一个方法


|     | counter | family | flower | garden | horns | horse | room | train | trex | truck |
| --- | ------- | ------ | ------ | ------ | ----- | ----- | ---- | ----- | ---- | ----- |
|     |         |        | 11     | 14     |       | 37    |      | 11    | 38   | 2     |
|     |         |        |        | 4      |       |       |      |       |      |       |
|     |         |        |        |        |       |       |      |       |      |       |
|     |         |        |        |        |       |       |      |       |      |       |
|     |         |        |        |        |       |       |      |       |      |       |
|     |         |        |        |        |       |       |      |       |      |       |
|     |         |        |        |        |       |       |      |       |      |       |
|     |         |        |        |        |       |       |      |       |      |       |
|     |         |        |        |        |       |       |      |       |      |       |
|     |         |        |        |        |       |       |      |       |      |       |
|     |         |        |        |        |       |       |      |       |      |       |
|     |         |        |        |        |       |       |      |       |      |       |
|     |         |        |        |        |       |       |      |       |      |       |
|     |         |        |        |        |       |       |      |       |      |       |
|     |         |        |        |        |       |       |      |       |      |       |

11
14
4
37
**11**
38
**2**
33
3
30
24


|     | building | bird | city | deer | house | lion | mouse | town |     |
| --- | -------- | ---- | ---- | ---- | ----- | ---- | ----- | ---- | --- |
|     | 33       | 3    | 24   | 2    | x     |      | 20    | 20   |     |
|     |          | 30   |      |      |       |      |       |      |     |

deer  38
mouse 14
mouse 38
lion 38

34






\begin{figure*}[t]
\centering
\setlength{\tabcolsep}{0pt}
\renewcommand{\arraystretch}{0}

% ================= adjustable parameters =================
\newcommand{\cmpW}{0.185\linewidth}      % each image width
\newcommand{\cmpH}{0.105\linewidth}      % each image height
\newcommand{\colGap}{1.5pt}              % gap between columns
\newcommand{\rowGap}{1pt}                % gap between two views / content-style
\newcommand{\headGap}{2pt}               % gap between header and images
\newcommand{\caseGap}{2pt}               % gap around dashed separator
\newcommand{\dashW}{0.965\linewidth}     % dashed separator width
% ==========================================================

\newcommand{\cmpImg}[1]{%
\includegraphics[width=\cmpW,height=\cmpH]{#1}%
}

% left column: content + style
\newcommand{\cmpCS}[2]{%
\begin{tabular}{@{}c@{}}
\cmpImg{#1}\\[\rowGap]
\cmpImg{#2}
\end{tabular}%
}

% method column: two rendered views
\newcommand{\cmpViews}[2]{%
\begin{tabular}{@{}c@{}}
\cmpImg{#1}\\[\rowGap]
\cmpImg{#2}
\end{tabular}%
}

% one case: content/style + four methods
\newcommand{\cmpCase}[6]{%
\cmpCS{#1}{#2} &
#3 &
#4 &
#5 &
#6%
}

\newcommand{\cmpDash}{%
\\[\caseGap]
\multicolumn{5}{c}{%
\begin{tikzpicture}
\draw[dashed, line width=0.35pt] (0,0) -- (\dashW,0);
\end{tikzpicture}}\\[\caseGap]
}

\begin{tabular}{@{}c@{\hspace{\colGap}}c@{\hspace{\colGap}}c@{\hspace{\colGap}}c@{\hspace{\colGap}}c@{}}
{\scriptsize Content \& Style} &
{\scriptsize StyleGaussian} &
{\scriptsize StylizedGS} &
{\scriptsize SGSST} &
{\scriptsize Ours} \\[\headGap]

% ================= Case 1 =================
\cmpCase
  {truck.png}
  {tree.png}
  {\cmpViews{figure3_truck_tree_stylegaussian.png}{figure3_truck_tree_stylegaussian_view2.png}}
  {\cmpViews{figure3_truck_tree_stylizedgs.png}{figure3_truck_tree_stylizedgs_view2.png}}
  {\cmpViews{figure3_truck_tree_sgsst.png}{figure3_truck_tree_sgsst_view2.png}}
  {\cmpViews{figure3_truck_tree_proposed.png}{figure3_truck_tree_proposed_view2.png}}

\cmpDash

% ================= Case 2 =================
\cmpCase
  {train.png}
  {bamboo.png}
  {\cmpViews{figure3_train_bamboo_stylegaussian.png}{figure3_train_bamboo_stylegaussian_view2.png}}
  {\cmpViews{figure3_train_bamboo_stylizedgs.png}{figure3_train_bamboo_stylizedgs_view2.png}}
  {\cmpViews{figure3_train_bamboo_sgsst.png}{figure3_train_bamboo_sgsst_view2.png}}
  {\cmpViews{figure3_train_bamboo_proposed.png}{figure3_train_bamboo_proposed_view2.png}}

\cmpDash

% ================= Case 3 =================
\cmpCase
  {garden.png}
  {fall.png}
  {\cmpViews{figure3_garden_fall_stylegaussian.png}{figure3_garden_fall_stylegaussian_view2.png}}
  {\cmpViews{figure3_garden_fall_stylizedgs.png}{figure3_garden_fall_stylizedgs_view2.png}}
  {\cmpViews{figure3_garden_fall_sgsst.png}{figure3_garden_fall_sgsst_view2.png}}
  {\cmpViews{figure3_garden_fall_proposed.png}{figure3_garden_fall_proposed_view2.png}}
  
  \cmpDash

% ================= Case 4 =================
\cmpCase
  {horse.png}
  {grass.png}
  {\cmpViews{figure3_horse_grass_stylegaussian.png}{figure3_horse_grass_stylegaussian_view2.png}}
  {\cmpViews{figure3_horse_grass_stylizedgs.png}{figure3_horse_grass_stylizedgs_view2.png}}
  {\cmpViews{figure3_horse_grass_sgsst.png}{figure3_horse_grass_sgsst_view2.png}}
  {\cmpViews{figure3_horse_grass_proposed.png}{figure3_horse_grass_proposed_view2.png}}
  
  \cmpDash

% ================= Case 5 =================
\cmpCase
  {horns.png}
  {style3.png}
  {\cmpViews{figure3_horns_style3_stylegaussian.png}{figure3_horns_style3_stylegaussian_view2.png}}
  {\cmpViews{figure3_horns_style3_stylizedgs.png}{figure3_horns_style3_stylizedgs_view2.png}}
  {\cmpViews{figure3_horns_style3_sgsst.png}{figure3_horns_style3_sgsst_view2.png}}
  {\cmpViews{figure3_horns_style3_proposed.png}{figure3_horns_style3_proposed_view2.png}}
  
 

\end{tabular}

\caption{Qualitative comparison with representative 3DGS stylisation methods. Each example contains one content image and one style reference, followed by stylised renderings from two novel viewpoints.}
\label{fig:qualitative}
\end{figure*}



method     voting    ratio  percent
   方法1   9.916667 0.041319     4.13
   方法2   3.083333 0.012847     1.28
   方法3  77.083333 0.321181    32.12
   方法4 149.916667 0.624653    62.47



Figure 1 and Figure 3 have been typeset as LaTeX tables. For Figure 3, I added two more examples, and each example now includes one additional view. The first view of all four methods includes zoomed-in regions, while the second view is shown without zoom-in.

For the user study, I treated the method receiving the highest score for each evaluated instance as receiving one vote, and summarised the voting results in "tab:user_study". I also added a short description of this voting at the end of the user study paragraph.


The reason is that the ablation study focuses on the specific effects of our components, rather than the temporal consistency evaluated in Table 1. LPIPS/RMSE mainly measure cross-view consistency, while our three components mainly affect stylisation quality, content preservation, and colour/style alignment. Therefore, we kept ArtFID for overall quality and added CLIP-C/CLIP-S to better reflect the effects of the edge and chroma losses.

I will update Fig. 3 later today by using different zoomed-in regions for the second view to reduce redundancy between the two views.



