
\section{Introduction}
\label{introduction}
3D scene stylization aims to transfer the artistic appearance of a reference image to a 3D scene while preserving plausible rendering under novel viewpoints. With broad applications in artistic content creation, gaming, virtual reality, and digital entertainment, this problem has attracted increasing attention in recent years. Earlier studies have explored stylization on meshes~\citep{kato2018neural}, point clouds, and neural radiance fields (NeRF) \citep{huang2022stylizednerf}, demonstrating the feasibility of extending style transfer from 2D images to 3D representations \citep{hollein2022stylemesh,cao2020psnet,zhang2022arf}. However, these representations often suffer from limitations in rendering efficiency, view consistency, or practical usability, which still hinder broader deployment in interactive 3D content creation. 

Recently, 3D Gaussian Splatting (3DGS) \citep{kerbl20233d} has emerged as a powerful 3D representation for novel-view synthesis, thanks to its strong reconstruction quality, efficient optimization, and real-time rendering capability. Building on these advantages, recent representative 3DGS-based stylization methods have substantially improved the visual quality and multi-view consistency of stylized 3D scenes. Despite these advances, existing methods still rely on a pre-reconstructed 3DGS scene obtained from posed multi-view observations, or equivalently assume that dense multi-view inputs and camera poses are already available. Such requirements are restrictive in practical usage, where casual users are often unable to capture a full multi-view sequence of a scene and instead prefer a much simpler input setting, namely, a single content image together with a style reference image.


\begin{figure}[t]
\centering
% \makebox[\textwidth][c]{
    \includegraphics[width=\linewidth]{./Figure_0.png}
% }

\caption{Illustration of two challenges in single-image generative 3DGS stylization. (a) Direct optimization of bounded RGB colors can lead to color drift during long-iteration stylization. (b) Under low-chroma style references, the stylized results are more prone to chromatic instability and spurious color artifacts.}

\label{introduce_problem}
\end{figure}


Fortunately, recent advances in camera-controllable video diffusion~\citep{voleti2024sv3d,xu2024cavia} and feed-forward generative 3D content creation~\citep{bahmani2025lyra,tang2024lgm} provide a new path toward this goal. In particular, single-image generative 3DGS models make it possible to directly obtain a view-consistent 3D Gaussian scene from only one image~\citep{bahmani2025lyra,roh2025catsplat}, thereby opening a new setting for 3D style transfer with substantially lower content acquisition cost. However, directly extending existing optimization-based 3DGS stylization methods to this setting remains non-trivial. On the one hand, in our setting Gaussian appearance is explicitly represented by bounded RGB values, and directly optimizing these exported colors over long iterations can induce deviations from the valid bounded RGB domain, leading to noticeable color drift, such as spurious color patches and local color inconsistency, as illustrated in \autoref{introduce_problem}(a). On the other hand, the single-image generative setting makes it more difficult to preserve local structural details and maintain chromatic stability; as shown in \autoref{introduce_problem}(b), the latter issue becomes particularly noticeable under style references with weak or highly restricted chromatic cues, where undesired blue/orange tints may appear in visually neutral regions.



To address these challenges, we propose a single-image generative stylization framework for 3D Gaussian Splatting, which extends optimization-based 3DGS stylization to a more practical single-image setting. We first employ a camera-conditioned video diffusion process together with a 3DGS decoder to construct an initial view-consistent Gaussian scene from a single content image, thereby enabling scene-level stylization without requiring captured multi-view observations. Starting from this generated initialization, we further propose a latent-space color reparameterization tailored to the bounded RGB parameterization of generated Gaussians, enabling Gaussian colors to be optimized in an unconstrained latent space and mapped back to valid RGB values for rendering, thereby suppressing color drift caused by violations of the valid RGB bounds during long-iteration optimization. Furthermore, we design an edge-consistency loss to better preserve local structural details, together with a chroma-consistency loss to suppress undesired chromatic deviation, particularly for low-chroma style references. In this way, our method enables practical single-image 3D scene stylization while improving optimization reliability and rendering fidelity in the generative 3DGS setting.

The main contributions of this paper are summarized as follows:
\begin{itemize}
    \item A single-image generative stylization framework for 3D Gaussian Splatting, which initializes the scene from a generated 3DGS representation and performs stylization under differentiable rendering, extending optimization-based 3DGS stylization from reconstructed scenes to a more practical single-image setting.  To the best of our knowledge, this is the first work to extend optimization-based 3DGS stylization to a single-image generative 3DGS initialization setting. 
    \item  A latent-space color reparameterization tailored to the bounded RGB parameterization of generative 3DGS, enabling stable long-iteration optimization by alleviating color drift caused by deviations from the bounded RGB domain. 
    \item Edge-consistency and chroma-consistency regularizations for the generative 3DGS setting, which improve local structural preservation and chromatic stability during stylization. 
\end{itemize}



We organize the paper as follows. Section~\ref{related_work} reviews related work on 3D scene stylization, 3D Gaussian Splatting, and generative 3D scene reconstruction. Section~\ref{method} presents the proposed single-image generative 3DGS stylization framework in detail. Section~\ref{experiments} reports the experimental results, including comparisons with existing methods and ablation studies. Finally, Section~\ref{conclusion} concludes the paper and discusses future directions.


\section{Related work}
\label{related_work}

\subsection{Image Style Transfer}
\label{image_style_transfer}
Image style transfer aims to synthesize an image that preserves the content structure of one image while adopting the artistic characteristics of another. \cite{gatys2016image} first formulate style transfer as an optimization problem in a pretrained CNN feature space, where content is preserved by feature reconstruction and style is captured by Gram-matrix statistics. Subsequent works  introduce feed-forward networks for fast stylization of fixed styles~\citep{johnson2016perceptual,ulyanov2016instance}. Later, arbitrary style transfer methods such as AdaIN~\citep{huang2017arbitrary} and WCT~\citep{li2017universal} further improve flexibility by aligning feature statistics without retraining a separate model for each style. More recently, Transformer-based~\citep{deng2022stytr2}, GAN-based~\citep{zhu2017unpaired,xu2021drb}, and diffusion-based~\citep{zhang2023inversion,chung2024style,deng2024z} approaches have further advanced semantic correspondence, stylistic controllability, and visual realism. These image style transfer methods provide important perceptual objectives and architectural inspiration for later 3D style transfer.


% \subsection{3D Gaussian Splatting}
% \label{3d_gaussian_splatting}
% 3D Gaussian Splatting (3DGS) \citep{kerbl20233d} has emerged as a high-fidelity alternative to implicit representations for 3D scene modeling. In 3DGS, a scene is represented as a massive collection of anisotropic 3D Gaussians. Each Gaussian is defined by a center position $\mu \in \mathbb{R}^3$, a covariance matrix $\Sigma$ (derived from scaling $S$ and rotation $R$), an opacity $\alpha \in [0,1]$, and color coefficients $c$ typically represented by Spherical Harmonics (SH). The rendering process employs a tile-based differentiable rasterizer, where the color $C$ of a pixel is computed by $\alpha$-blending $N$ ordered Gaussians overlapping the pixel:
% \begin{equation}
% C = \sum_{i \in N} c_i \alpha_i \prod_{j=1}^{i-1} (1 - \alpha_j),
% \end{equation}
% where $\alpha_i$ is the density of the $i$-th Gaussian at the pixel, computed as the product of its learned opacity and the evaluated 2D Gaussian distribution. Beyond static reconstruction, the field is increasingly shifting toward generative 3DGS. For instance, Lyra \citep{bahmani2025lyra} leverages Gaussian priors within a generative framework to enable high-quality 3D content creation, providing a more flexible backbone for downstream tasks like style transfer.

\subsection{3D Style Transfer}
\label{3d_style_transfer}
3D style transfer extends conventional 2D stylization to 3D scene representations and aims to produce stylized outputs that remain consistent across viewpoints. Early efforts were explored on meshes and point clouds, showing that artistic style can be transferred beyond 2D images to explicit 3D representations~\citep{kato2018neural,hollein2022stylemesh,cao2020psnet}. Later works mainly focus on Neural Radiance Fields (NeRF)~\citep{mildenhall2021nerf}. ARF~\citep{zhang2022arf} introduces nearest-neighbor feature matching for radiance-field stylization and improves detail preservation over Gram-based formulations. Subsequent NeRF-based methods further perform stylization in radiance-field feature space and strengthen multi-view consistency through explicit scene-aware designs~\citep{liu2023stylerf,meric2024g3dst}.

Recent contributions suggest that 3D Gaussian Splatting is a promising representation for 3D scene stylization. StyleGaussian~\citep{liu2024stylegaussian} proposes an instant feed-forward pipeline that embeds 2D VGG features into reconstructed Gaussians and decodes stylized RGB while preserving strict multi-view consistency. G-Style~\citep{kovacs2024gstyle} follows an optimization-based formulation and explicitly modifies Gaussian attributes to enhance stylization quality. InstantStyleGaussian~\citep{yu2024instantstylegaussian} combines diffusion-generated style images with an iterative dataset update strategy to accelerate stylization on pre-reconstructed Gaussian scenes. StyleSplat~\citep{jain2024stylesplat} extends Gaussian-based stylization to localized object transfer by aligning spherical-harmonic coefficients for selected objects in a scene. SGSST~\citep{galerne2025sgsst} formulates 3DGS stylization as an optimization problem on pretrained scenes and introduces a multiscale SOS loss for high-resolution and ultra-high-resolution stylization. More recent works begin to explore feed-forward 3DGS stylization, moving beyond scene-specific optimization toward faster and more scalable pipelines~\citep{du2025optimization,liu2025stylos,kaleta2026anystyle}.

\subsection{Generative 3D scene reconstruction}
\label{generative_3d_scene_reconstruction}
Recent progress in generative and feed-forward 3D reconstruction has significantly reduced the input burden of 3D content creation. Early Gaussian-based generation methods such as DreamGaussian~\citep{tang2023dreamgaussian} and LGM~\citep{tang2024lgm} show that explicit Gaussian representations can support efficient 3D asset generation from text prompts or single images. Subsequent reconstruction-oriented methods, including InstantMesh~\citep{xu2024instantmesh}, CATSplat~\citep{roh2025catsplat}, GS-LRM~\citep{zhang2024gs}, and AnySplat~\citep{jiang2025anysplat}, further improve the practicality of reconstructing 3D Gaussian scenes from single, sparse-view, or pose-free image inputs. More recent methods such as UniForward~\citep{tian2025uniforward} and Lyra~\citep{bahmani2025lyra} continue this trend by enabling feed-forward generation of view-consistent 3DGS scenes from sparse inputs or even a single image. Building on these advances in practical 3DGS initialization, our work studies how a generated 3DGS scene can be further stylized under differentiable rendering.



\section{Method}
\label{method}
In this section, we present our method for stylizing a single-image initialized generative 3D Gaussian Splatting (3DGS) scene. We first provide an overview of the problem setting and the overall pipeline, as illustrated in \autoref{framework}. We then describe the single-image generative 3DGS initialization, followed by the appearance optimization strategy and the rendering-domain stylization objectives in detail.

\begin{figure}[!t]
  \centering
  \includegraphics[width=0.95\linewidth]{./Figure_1.png}
  \caption{Overview of our framework.}
  \label{framework}  
\end{figure}

\subsection{Overview}
\label{overview}
Given a single content image $I_c$ and a reference style image $I_s$, our goal is to obtain a stylized 3D Gaussian Splatting (3DGS) scene $\mathcal{G}^*$ that supports view-consistent novel-view rendering. Different from existing optimization-based 3DGS style transfer methods, which typically assume that a realistic 3D scene has already been reconstructed from multi-view captured images, we consider a more practical single-image setting.  Our method consists of two stages. First, a single-image generative pipeline \citep{bahmani2025lyra} produces both view-consistent content targets and an initial 3DGS scene from $I_c$. Second, starting from this generated initialization, we optimize scene appearance under differentiable rendering~\citep{kerbl20233d} using a multiscale style-transfer objective \citep{galerne2025sgsst}. To make this optimization compatible with the bounded RGB parameterization of the generated 3DGS, we perform appearance optimization in an unconstrained latent color space via a reparameterization strategy and further regularize the rendered results with additional edge and chroma constraints. 


\subsection{Single-image generative 3DGS initialization}
\label{singleimg}
Different from conventional optimization-based 3DGS stylization, which starts from a reconstructed 3D scene and captured multiview observations, our method is initialized from only a single content image $I_c$. To make stylization feasible in this setting, we build on recent advances in camera-controllable video diffusion models \citep{wang2024motionctrl,he2024cameractrl,ren2025gen3c} and feed-forward generative 3DGS reconstruction \citep{bahmani2025lyra}. Given the input image $I_c$ and a predefined set of target viewpoints $\Pi=\{\pi_t\}_{t=1}^{T}$, a camera-conditioned video diffusion prior, denoted by $F_{\mathrm{gen}}$, first produces a view-consistent latent sequence

\begin{equation} 
\mathcal{Z}=\{z_t\}_{t=1}^{T}=F_{\mathrm{gen}}(I_c,\Pi).
\end{equation}

These latent observations are then decoded along two branches. First, an RGB decoder, denoted by $\mathcal{D}_{\mathrm{rgb}}$, produces a set of content targets
\begin{equation} 
 I_t = \mathcal{D}_{\mathrm{rgb}}(z_t), \qquad t=1,\dots,T
\end{equation} 

These generated images provide rendering-domain content supervision during stylization. Second, a 3DGS decoder, denoted by $\mathcal{D}_{\mathrm{gs}}$, predicts an explicit initial 3DGS scene \begin{equation}
\mathcal{G}_0 = \mathcal{D}_{\mathrm{gs}}(\mathcal{Z}),
\end{equation}

which serves as the starting point of the subsequent optimization. In this way, a single input image gives rise to both the scene representation to be stylized and a view-consistent set of generated supervision targets.

Formally, let $\mathcal{G}_0=\{g_i^0\}_{i=1}^{N}$ denote the initial Gaussian set. Each Gaussian is parameterized as
\begin{equation}
g_i^0=\left(\mathbf{p}_i^0,\, o_i^0,\, \mathbf{s}_i^0,\, \mathbf{q}_i^0,\, \mathbf{c}_i^0\right).
\end{equation}

Here, $\mathbf{p}_i^0 \in \mathbb{R}^3$ is the Gaussian center, $o_i^0 \in [0,1]$ denotes the opacity, $\mathbf{s}_i^0 \in \mathbb{R}_{+}^{3}$ is the anisotropic scale, $\mathbf{q}_i^0 \in \mathbb{R}^{4}$ is a normalized rotation quaternion, and $\mathbf{c}_i^0 \in [0,1]^3$ is the RGB color.

The RGB-decoded observations are organized as $\mathcal{T}=\{(I_t,\pi_t)\}_{t=1}^{T}$, where $I_t$ is the generated target image at viewpoint $\pi_t$. During stylization, we render the current Gaussian scene under a sampled viewpoint $\pi_t$ and use the corresponding $I_t$ as the content target. Although the entire pipeline is initialized from a single image, the optimization is still guided by an internally consistent set of generated multiview-like targets, which is important for maintaining view coherence under novel-view rendering.

\subsection{Stable appearance optimization for generative 3DGS}
\label{optimization}

To preserve the generated scene geometry during stylization, we keep geometry-related Gaussian attributes fixed and optimize only the appearance variables \citep{galerne2025sgsst,zhuang2025styleme3d,jain2024stylesplat}. Specifically, we decompose the initialized scene as $\mathcal{G}_0 = [\Theta_{\mathrm{geo}}^{0}, \mathbf{C}^{0}],$ where $\Theta_{\mathrm{geo}}^{0}$ denotes the geometry-related attributes and $\mathbf{C}^{0}$ denotes the initialized Gaussian colors.  Accordingly, the stylized scene is written as $\mathcal{G}(\mathbf{C}) = [\Theta_{\mathrm{geo}}^{0}, \mathbf{C}]$.

A key difference between our setting and conventional optimization-based 3DGS stylization lies in the appearance parameterization. In reconstructed 3DGS scenes, Gaussian appearance is commonly represented by spherical-harmonic-based color features \citep{kerbl20233d}, and stylization is typically performed by updating SH-based appearance parameters rather than an explicit bounded RGB tensor. In contrast, the generated 3DGS in our pipeline represents Gaussian appearance explicitly as a three-channel RGB value, with initialized colors $\mathbf{C}^{0}\in[0,1]^{N\times 3}$. Under this bounded parameterization, directly treating exported RGB values as unconstrained optimization variables is less compatible with the native generative representation and becomes less suitable for long-iteration optimization. In practice, the color updates can be gradually driven toward the boundary of the valid range, which often leads to color drifting artifacts. 

To alleviate this issue, we reparameterize Gaussian colors into an unconstrained latent space and optimize the latent variable instead. Specifically, starting from the initialized colors $\mathbf{C}^{0}$, we define
\begin{equation}
\mathbf{Z}^{p}_{0}=\phi^{-1}(\mathbf{C}^{0})=\operatorname{atanh}(2\mathbf{C}^{0}-1)
\end{equation}
where $\phi^{-1}(\cdot)$ maps bounded RGB values to an unconstrained latent space. During optimization, the current Gaussian colors are recovered through
\begin{equation}
\mathbf{C}(\mathbf{Z}^{p})=\phi(\mathbf{Z}^{p})=\frac{1}{2}\tanh(\mathbf{Z}^{p})+\frac{1}{2}
\end{equation}

Accordingly, the stylized scene is written as
\begin{equation}
\mathcal{G}(\mathbf{Z}^{p})=\left[\Theta_{\mathrm{geo}}^{0},\,\mathbf{C}(\mathbf{Z}^{p})\right]
\end{equation}

We then formulate appearance optimization as
\begin{equation}
\min_{\mathbf{Z}^{p}\in\mathbb{R}^{N\times 3}}
\mathbb{E}_{(I_t,\pi_t)\sim\mathcal{T}}
\Big[
\mathcal{L}_{\mathrm{app}}
\big(
\hat{I}_{t}(\mathbf{Z}^{p}),\, I_t,\, I_s
\big)
\Big]
\end{equation}

where $\mathcal{L}_{\mathrm{app}}$ is the rendering-domain stylization objective defined in Section~\ref{loss}, and $\hat{I}_ t$ is the rendered image of the current Gaussian scene under viewpoint $\pi_t$. This constrained-to-unconstrained reformulation turns the appearance update into a smoother optimization problem, while still ensuring that the rendered Gaussian colors remain valid. In practice, it improves the stability of long-iteration optimization and reduces color drifting artifacts.

\subsection{Rendering-domain style transfer loss}
\label{loss}
Based on the initialization and appearance parameterization described above, for a sampled training pair $(I_t,\pi_t)\sim\mathcal{T}$, we optimize the stylized scene with the following rendering-domain style transfer loss:
\begin{equation}
\mathcal{L}_{\mathrm{app}}
=
\mathcal{L}_{\mathrm{ms}}
+\lambda_{e}\mathcal{L}_{\mathrm{edge}}
+\lambda_{c}\mathcal{L}_{\mathrm{chr}}
\end{equation}
Here, $\mathcal{L}_{\mathrm{ms}}$ is the primary multiscale VGG-based style transfer term, while $\mathcal{L}_{\mathrm{edge}}$ and $\mathcal{L}_{\mathrm{chr}}$ are two additional regularization terms for structure preservation and chroma stabilization, respectively.

\subsubsection{Multiscale style supervision}
We adopt a multiscale VGG-based style transfer loss \citep{galerne2025sgsst} as the primary stylization objective. Specifically, both the content term and the style term are computed in VGG feature space, where the style term matches multilevel feature statistics including Gram matrices \citep{gatys2016image}, feature-wise means, and standard deviations \citep{huang2017arbitrary}. These VGG-based supervision terms are further imposed across multiple image scales, yielding
\begin{equation}
\mathcal{L}_{\mathrm{ms}}
=
\sum_{s\in\mathcal{S}} w_s
\left(
\mathcal{L}_{\mathrm{sty}}\bigl(\hat{I}_t^{(s)}, I_s^{(s)}\bigr)
+\lambda_{\mathrm{cnt}}
\mathcal{L}_{\mathrm{cnt}}\bigl(\hat{I}_t^{(s)}, I_t^{(s)}\bigr)
\right)
\end{equation}
where $\mathcal{S}$ denotes the set of active scales, $w_s$ is the weight associated with scale $s$, and $\hat{I} _t ^{(s)}$, $I _t^{(s)}$, and $I_s^{(s)}$ denote the rendered result, the generated content target, and the style reference resized to scale $s$, respectively. In practice, we first optimize the coarsest active scale and then switch to joint optimization over all active scales \citep{galerne2025sgsst}.

\subsubsection{Edge-aware structure preservation}
Although the multiscale VGG objective already includes a content term, it does not explicitly constrain local edge consistency in the image domain. This issue is more noticeable in our single-image generative setting, where the stylization process is guided by generated content targets rather than captured multiview observations. To better preserve scene contours and local structural details, we introduce an additional edge-preservation term directly on the rendered result.

Specifically, we first convert the rendered image and the corresponding generated content target into luminance maps, and then extract their Sobel edge responses. The edge loss is defined as
\begin{equation}
\mathcal{L}_{\mathrm{edge}}
=
\left\|
\operatorname{Sobel}\!\bigl(Y(\hat{I}_t)\bigr)
-\operatorname{Sobel}\!\bigl(Y(I_t)\bigr)
\right\|_{1}
\end{equation}
where $Y(\cdot)$ denotes luminance conversion. This term encourages the stylized rendering to remain aligned with the local structural boundaries of the content target, thereby alleviating the loss of fine structural details under strong stylization.

\subsubsection{Style-guided chroma stabilization}
In our generative 3DGS stylization setting, we observe that the rendered results are more susceptible to chromatic drift, especially when the reference style exhibits weak chromatic cues, such as low-saturation or near-achromatic artworks. A possible reason is the difference in appearance parameterization between our setting and conventional reconstructed-scene 3DGS stylization: we optimize explicit RGB colors, whereas prior methods typically operate on SH-based appearance coefficients. To improve chromatic stability during stylization, we introduce a style-guided chroma regularizer in CIE Lab space. The chroma loss is defined as
\begin{equation}
\mathcal{L}_{\mathrm{chr}}=
\left\|
\mathbf{s}_{ab}(\hat{I}_t)
-\mathbf{s}_{ab}^{\tau}(I_s)
\right\|_1 
\end{equation}

where $\mathbf{s}_{ab}(I)$ denotes the mean and standard deviation of the $ab$ channels in Lab space, and $\tau$ denotes trimmed statistics~\citep{wilcox2012introduction} on the style image . This term regularizes the global chromatic statistics of the rendered result toward the style reference, which is useful for suppressing spurious color artifacts in low-chroma stylization cases.



\section{Experiments}
\label{experiments}

\subsection{Implementation Details}
For comparative evaluation,  we conduct experiments on scenes  selected from the Mip-NeRF 360 \citep{barron2022mip}, Tanks\&Temples \citep{knapitsch2017tanks}, and LLFF \citep{mildenhall2019local} datasets. For each scene, we select one image as the content input and use Lyra \citep{bahmani2025lyra} to generate the initial 3DGS scene. Unless otherwise specified, all experiments are conducted at a rendering resolution of 704 $\times$ 1280. We optimize each scene-style pair for 20,000 iterations using the Adam optimizer with a learning rate of 0.0025.

We adopt two scales for the multiscale stylization objective, namely a half-resolution scale and the original rendering scale, and assign them equal weights. In practice,  we optimize only the coarsest scale during the first 10,000 iterations and then jointly optimize both scales during the remaining 10,000 iterations. The weight of the content term is set to 5.0. For the additional regularization terms, the weights of the edge-consistency loss and chroma-consistency loss are set to 0.05 and 0.03, respectively, and the trimming ratio used for style-side chroma statistics is set to 0.05. For fair comparison, all methods are evaluated under matched output resolutions and camera trajectories. 

In implementation, the gradients of the main multiscale style term and the two auxiliary regularization terms are combined by a gradient-balancing strategy, and PCGrad~\citep{yu2020gradient} is used to alleviate gradient conflicts between different objectives. Our implementation is based on PyTorch, and all experiments are conducted on a single NVIDIA L40S GPU. 




\subsection{Comparison with existing methods}
\subsubsection{Qualitative comparison}

\begin{figure}[htbp]
\centering
\includegraphics[width=\linewidth]{./Figure_2.png}
\caption{Qualitative comparison with representative 3DGS stylization methods. The first column shows the content and style images, and the remaining columns show stylized novel-view renderings produced by different methods.}
\label{fig:qualitative}
\end{figure} 

~\autoref{fig:qualitative} compares our method with representative 3DGS stylization methods. StyleGaussian \citep{liu2024stylegaussian} mainly captures the global color tendency of the reference style, while its fine-scale texture transfer remains limited; this is particularly visible in the third row, where the vegetation and ground are simplified into broad color patches rather than layered brushstroke patterns.  StylizedGS \citep{zhang2025stylizedgs} can correctly transfer the overall style pattern, but its results sometimes exhibit noticeable color deviation from the reference style, leaving some regions with biased tones despite plausible stylization. SGSST \citep{galerne2025sgsst}, driven by multiscale style optimization, yields stronger painterly stylization, but under common rendering resolutions it sometimes shows softened boundaries and less stable color rendition. In comparison, our method achieves similarly strong stylization while better preserving structural clarity and chromatic stability. As highlighted by the boxed zoom-in regions in Fig.~\autoref{fig:qualitative}, our results retain sharper and more legible trailer boundaries in the truck scene than SGSST. Likewise, in the train scene, the enlarged comparison of the rear section shows that our method preserves the color appearance more faithfully, while SGSST exhibits a more noticeable color shift in that region. Overall, our method provides a better trade-off between style expressiveness, structural preservation, and color stability, while extending optimization-based 3DGS stylization to single-image initialized generative 3DGS scenes.

 


\subsubsection{Quantitative comparison}
Following recent 3D scene stylization studies, we evaluate both stylization quality and cross-view texture consistency. Following the rendering setup of \cite{bahmani2025lyra}, each scene-style example is rendered from 6 trajectories with 31 frames per trajectory, and we randomly sample 8 frames from each example to compute the style-related metrics, while using all 186 rendered frames for the consistency evaluation. Style fidelity is assessed by Gram loss and ArtFID. Specifically, Gram loss measures the discrepancy between the multi-level style statistics of the stylized renderings and the reference style image, while ArtFID evaluates the overall stylization quality from a distributional perspective. For texture consistency, we follow prior view-warping-based evaluation protocols \citep{lai2018learning, ruder2018artistic} and report warped LPIPS \citep{zhang2018unreasonable} and RMSE \citep{mildenhall2019local} on rendered view pairs at both short-range ($i{=}1$) and long-range ($i{=}10$) intervals. Concretely, we estimate optical flow on the original rendered sequence, use it to warp the stylized frame at time $t$ to frame $t+i$, and then compute LPIPS and RMSE between the warped result and the stylized frame at $t+i$ over valid warped regions only, with $i{=}1$ and $i{=}10$ corresponding to short-range and longer-range consistency, respectively. The pairwise scores are first averaged over all valid frame pairs within each rendered trajectory, then over the 6 trajectories of each example, and finally over all 90 scene-style pairs. Lower values indicate better performance for all metrics.

Quantitative results are reported in Table~\ref{tab:quantitative}. The image stylization methods, including AdaIN, AesPA-Net, and SaMam, can produce plausible stylization on individual rendered frames, but exhibit relatively weak temporal consistency due to the lack of explicit inter-frame constraints. The video stylization methods, such as ReReVST, CCPL, and CAP-VSTNet, improve continuity by enforcing temporal coherence on rendered frame sequences, and therefore outperform image-based methods on consistency. Notably, their results are particularly competitive at the larger interval $i{=}10$, showing that 2D temporal regularization remains effective for longer-gap comparisons on rendered videos. However, their temporal consistency is still generally inferior to ours, since their consistency is imposed after rendering rather than directly at the scene representation level. Among the 3D stylization baselines, our method achieves the best Gram loss and ArtFID scores while maintaining strong cross-view consistency, demonstrating the best overall balance between stylization quality and consistency.

\begin{table}[t]
\caption{Quantitative comparison with representative style transfer methods on 90 scene-style pairs. Lower is better for all metrics.}
\vspace{4pt}
\label{tab:quantitative}
\centering
\resizebox{\linewidth}{!}{
\begin{tabular}{lcccccc}
\toprule
Method & Gram $\downarrow$ & ArtFID $\downarrow$ & LPIPS (i=1) $\downarrow$ & RMSE (i=1) $\downarrow$ & LPIPS (i=10) $\downarrow$ & RMSE (i=10) $\downarrow$ \\
\midrule
AdaIN             &\textbf{0.43e8}      &12.18     &0.151    &0.071     &0.267     &0.115  \\
AesPA-Net      &1.36e8      &14.74     &0.177    &0.068     &0.258     &0.099  \\
SaMam           &1.02e8      &15.61     &0.141    &0.074     &0.244     &0.117  \\
ReReVST         &1.95e8      &13.86     &0.078    &0.025     &0.180     &\textbf{0.061}  \\
CCPL              &\underline{0.49e8}      &15.41     &0.116    &0.057     &0.211     &0.098  \\
CAP-VSTNet  &0.68e8      &14.78     &0.073    &0.030     &0.174     &0.076  \\
StyleGaussian & 2.09e8 & 16.71 & \textbf{0.053} & \underline{0.024} & \underline{0.165} & 0.078 \\
SGSST         & 1.10e8 & \underline{11.60} & 0.062          & 0.028 & 0.192 & 0.089 \\
StylizedGS    & 3.17e8 & 14.91 & 0.070          & 0.027 & 0.243 & 0.083 \\
\midrule
Ours          & 0.91e8 & \textbf{10.37} & \underline{0.055} & \textbf{0.021} & \textbf{0.142} &\underline{0.067} \\
\bottomrule
\end{tabular}
}
\end{table} 
 

% \begin{table}[t]
% \caption{Quantitative comparison with representative 3DGS stylization methods on 90 scene-style pairs. Lower is better for all metrics.}
% \vspace{4pt}
% \label{tab:quantitative_3d}
% \centering
% \resizebox{\linewidth}{!}{
% \begin{tabular}{lcccccc}
% \toprule
% Method & Gram $\downarrow$ & ArtFID $\downarrow$ & LPIPS (i=1) $\downarrow$ & RMSE (i=1) $\downarrow$ & LPIPS (i=10) $\downarrow$ & RMSE (i=10) $\downarrow$ \\
% \midrule
% StyleGaussian & 2.09e8 & 16.71 & \textbf{0.053} & \underline{0.024} & \underline{0.165} & \underline{0.078} \\
% SGSST         & \underline{1.10e8} & \underline{11.60} & 0.062          & 0.028 & 0.192 & 0.089 \\
% StylizedGS    & 3.17e8 & 14.91 & 0.070          & 0.027 & 0.243 & 0.083 \\
% \midrule
% Ours          &  \textbf{0.91e8} & \textbf{10.37} & \underline{0.055} & \textbf{0.021} & \textbf{0.142} & \textbf{0.067} \\
% \bottomrule
% \end{tabular}
% }
% \end{table} 

% \begin{table}[t]
% \caption{Quantitative comparison with representative video stylization methods on 90 scene-style pairs. Lower is better for all metrics.}
% \vspace{4pt}
% \label{tab:quantitative_video}
% \centering
% \resizebox{\linewidth}{!}{
% \begin{tabular}{lcccccc}
% \toprule
% Method & Gram $\downarrow$ & ArtFID $\downarrow$ & LPIPS (i=1) $\downarrow$ & RMSE (i=1) $\downarrow$ & LPIPS (i=10) $\downarrow$ & RMSE (i=10) $\downarrow$ \\
% \midrule
% ReReVST         &1.95e8      &\underline{13.86}     &0.078    &\underline{0.025}     &0.180     &\textbf{0.061}  \\
% CCPL              & \textbf{0.49e8}      &15.41     &0.116    &0.057     &0.211     &0.098  \\
% CAP-VSTNet  &\underline{0.68e8}      &14.78     &\underline{0.073}    &0.030     &\underline{0.174}     &0.076  \\
% \midrule
% Ours          & 0.91e8 & \textbf{10.37} &  \textbf{0.055} & \textbf{0.021} & \textbf{0.142} &\underline{0.067} \\
% \bottomrule
% \end{tabular}
% }
% \end{table} 

% \begin{table}[t]
% \caption{Quantitative comparison with representative image stylization methods on 90 scene-style pairs. Lower is better for all metrics.}
% \vspace{4pt}
% \label{tab:quantitative_img}
% \centering
% \resizebox{\linewidth}{!}{
% \begin{tabular}{lcccccc}
% \toprule
% Method & Gram $\downarrow$ & ArtFID $\downarrow$ & LPIPS (i=1) $\downarrow$ & RMSE (i=1) $\downarrow$ & LPIPS (i=10) $\downarrow$ & RMSE (i=10) $\downarrow$ \\
% \midrule
% AdaIN             &\textbf{0.43e8}      &\underline{12.18}     &0.151    &0.071     &0.267     &0.115  \\
% AesPA-Net      &1.36e8      &14.74     &0.177    &\underline{0.068}     &0.258     &\underline{0.099}  \\
% SaMam           &1.02e8      &15.61     &\underline{0.141}    &0.074     &\underline{0.244}     &0.117  \\
% \midrule
% Ours          & \underline{0.91e8} & \textbf{10.37} &  \textbf{0.055} & \textbf{0.021} & \textbf{0.142} & \textbf{0.067} \\
% \bottomrule
% \end{tabular}
% }
% \end{table} 



\subsection{Perceptual study}
To further validate our approach, we conducted a user study with 24 participants. We randomly selected 40 scene-style pairs from the test set and compared our method against StyleGaussian \citep{liu2024stylegaussian}, StylizedGS \citep{zhang2025stylizedgs}, and SGSST \citep{galerne2025sgsst}. Each participant evaluated 10 randomly sampled instances. For each instance, the style image and the stylized videos rendered along the same camera trajectory by the four methods were presented in random order without revealing the method identity. Participants were asked to assess each result from two perspectives, namely style transfer quality and video consistency, and provide an overall score on a five-point Likert scale, where 1 denotes the worst and 5 denotes the best. For each method, we first averaged the scores across all evaluated instances for each participant, and then reported the mean and standard deviation across participants as the Mean Opinion Score (MOS). The results in ~\autoref{tab:user_study} show that our method achieves the highest average score, confirming superior perceptual quality.

\begin{table}[t]
\caption{User study results measured by Mean Opinion Score (MOS). Scores are reported as mean $\pm$ standard deviation across participants on a five-point Likert scale. Higher is better.}
\vspace{4pt}
\label{tab:user_study}
\centering
\resizebox{\linewidth}{!}{
\begin{tabular}{lcccc}
\toprule
Metric & StyleGaussian & StylizedGS & SGSST & Ours \\
\midrule
MOS $\uparrow$ & 1.85 $\pm$ 0.53 & 2.23 $\pm$ 0.60 & 3.91 $\pm$ 0.44 & \textbf{4.31 $\pm$ 0.46} \\
\bottomrule
\end{tabular}
}
\end{table}

\subsection{Ablation study}
To validate the contribution of each key component, we conduct ablation studies on the proposed latent-space optimization strategy and the two additional regularization terms introduced in Section~\ref{optimization} and Section~\ref{loss}.

\subsubsection{Influence of  latent-space optimization}

\begin{figure}[htbp]
  \centering
  \includegraphics[width=\linewidth]{./Figure_3.png}
  \caption{Ablation study on latent-space optimization.}
  \label{fig:ablation_param}  
\end{figure}


As shown in ~\autoref{fig:ablation_param}, directly optimizing Gaussian colors in RGB space leads to noticeable color drift during stylization, resulting in less faithful color rendition. In contrast, the proposed latent-space optimization effectively alleviates this issue and produces results with more stable and style-consistent color appearance. 


\subsubsection{Influence of edge-aware structure preservation}
\begin{figure}[htbp]
  \centering
  \includegraphics[width=\linewidth]{./Figure_4.png}
    \caption{Ablation study on edge-aware structure preservation.}
  \label{fig:ablation_edge}  
\end{figure}

~\autoref{fig:ablation_edge} shows the impact of our edge-aware structure preservation loss. The model without the edge-aware regularization leads to weaker preservation of local structural boundaries during stylization (e.g., the outlines of the mountain peaks in the second row appear blurrier). By contrast, with the proposed edge loss, the stylized results better preserve scene contours and local details, producing sharper and more structurally coherent renderings.



\subsubsection{Influence of style-guided chroma stabilization}
\begin{figure}[htbp]
  \centering
  \includegraphics[width=\linewidth]{./Figure_5.png}
  \caption{Ablation study on style-guided chroma stabilization.}
  \label{fig:ablation_chroma}  
\end{figure}
As shown in ~\autoref{fig:ablation_chroma}, without the chroma regularization, the stylized results are more prone to undesired color deviations, especially for nearly achromatic reference styles such as black-and-white artworks (e.g., the wings of the owls in the second row exhibit a bluish tint). In contrast, the proposed chroma regularization effectively suppresses such artifacts and produces more stable and visually consistent color rendition.



\subsection{Limitations}

While our method achieves appealing and style-consistent results in the generative 3DGS setting, several non-trivial limitations remain. First, the current pipeline is still computationally demanding. Our method relies on iterative scene-specific optimization driven by multiscale VGG statistics, which makes it more suitable for offline content creation than for real-time or interactive applications. Second, the proposed method may still struggle under severe content--style mismatch. Since the dominant stylization signal is still provided by global multiscale style statistics, highly abstract or structurally incompatible style references may lead to over-transferred textures or imperfect local structure preservation. Although the proposed edge-consistency and chroma-consistency regularizations improve structural and chromatic stability, they cannot fully resolve this issue, because the primary optimization objective remains global style matching.

In future work, we will focus on two directions. One is to improve computational efficiency through lighter optimization strategies or feed-forward stylization mechanisms. The other is to introduce more structure-aware or region-aware style constraints, so as to better handle challenging content--style combinations while better preserving fine scene details.





\section{Conclusion}
\label{conclusion}

In this paper, we propose a novel framework for 3D style transfer from a single input image, eliminating the reliance on multi-view observations required by existing methods. By leveraging generative 3DGS initialization together with diffusion-based priors, the proposed framework enables effective stylization under the challenging single-view setting. To address the incompatibility between generative and reconstructed 3DGS representations, we introduce a tailored optimization strategy that operates in a latent space, mitigating instability caused by bounded color parameterization. Moreover, by introducing edge regularization and chroma regularization, we alleviate the problems of insufficient content preservation and chromatic drift when existing methods are applied to generative 3DGS. Extensive experiments demonstrate that our method achieves strong stylization quality and consistent visual appearance, providing a practical solution for single-image 3DGS style transfer. In future work, we plan to extend the proposed framework to dynamic 4D scenes and explore feed-forward stylization models for improved efficiency and scalability.