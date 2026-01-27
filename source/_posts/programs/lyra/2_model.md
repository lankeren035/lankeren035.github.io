---
title: lyra模型
date: 2026-1-15 10:00:00
tags: [项目]
categories: [项目]
comment: true
toc: true



---

#

<!--more-->



## lyra模型

- 模型输入输出：
  - 输入图像分辨率： 704×1280 
  - 帧数：121
  - 压缩后latent（除以8）：88x160，帧数8
  - patch_size=2，token数量：56，320





## 3) LatentRecon 的详细网络结构拆解（含 QKV、shape）

下面我按 **Lyra 默认 720p 配置**（你也很接近：`enc_embed_dim=512, heads=8, mlp_ratio=4, patch_size=2, num_latent_c=16, latent_time_compression=8, latent_spat_compression=8, patch_size_out_factor=[1,8,8]`）来写 shape。

为避免你后续改结构时混乱，我同时给：

- **符号 shape**
- **一组典型数值（704×1280、T_out=121 ⇒ T_lat=16）**

------

# 3.1 输入张量与基础符号

- 目标图像分辨率：`H_img × W_img = 704 × 1280`
- VAE latent 空间压缩：`latent_spat_compression=8`
  - `H_lat = H_img/8 = 88`
  - `W_lat = W_img/8 = 160`
- 时间压缩：`latent_time_compression=8`
  - 若输出帧数（或“视角序列长度”）是 `T_out=121`
  - 则 latent 时间长度通常是 `T_lat = (T_out + 7)/8 = 16`
     -（这个关系也正好和后面的 ConvTranspose3d 输出长度公式对齐）
- patch 参数：
  - `patch_size_temporal = 1`
  - `patch_size = 2`
  - patch 后的网格：
    - `H_p = H_lat / 2 = 44`
    - `W_p = W_lat / 2 = 80`
  - token 数：
    - `N = T_lat * H_p * W_p = 16*44*80 = 56,320`
- 多视角并行（multi-view）：
  - `M = num_input_multi_views`（例如 6）
  - 编码时会出现两种 batch 形态：
    - **按 view 拆进 batch：** `B' = B*M`
    - **按 view 合并进序列：** `L = M*N`

------

# 3.2 模型结构树（Markdown）

## LatentRecon

### (A) Patch Embedding 三路输入（全部映射到 `embed_dim=512` 后相加）

#### 1) `patch_embed: PatchEmbed3D`

- **类型**：`Conv3d + LayerNorm`（flatten 成 tokens）
- **Conv3d**
  - `in_chans = num_latent_c = 16`
  - `out_chans = 512`
  - `kernel = (1,2,2)`, `stride=(1,2,2)`
- **输入**：`images_input_embed`
  - shape：`(B', T_lat, 16, H_lat, W_lat)`
  - 例：`(B*M, 16, 16, 88, 160)`
- **输出 tokens**
  - shape：`(B', N, 512)`，其中 `N=T_lat*H_p*W_p`
  - 例：`(B*M, 56,320, 512)`

#### 2) `patch_plucker_embed: PatchEmbed3D`（相机条件）

- **类型**：`Conv3d + LayerNorm`（zero_init=True，初始为“先不影响”）
- 若 `plucker_embedding_vae=true` 且 `fuse_type=concat`：
  - `in_chans = 2*num_latent_c = 32`
- **输入**：`plucker_embedding`
  - shape：`(B', T_lat, 32, H_lat, W_lat)`
- **输出**：`(B', N, 512)`
- **融合**：`x = x + plucker_tokens`

#### 3) `patch_time_embed` & `patch_time_embed_tgt`（时间条件）

- **类型**：两套 `PatchEmbed3D`（zero_init=True）
- `time_embedding_vae=true` 时：
  - `time_embedding_dim = num_latent_c = 16`
  - patch 同 `patch_embed`
- **输出**：都是 `(B', N, 512)`
- **融合**：`x = x + time_in + time_tgt`

------

### (B) 多视角融合（如果 `process_multi_views=true`）

- 当前 `x`：`(B', N, 512)`，其中 `B'=B*M`
- reshape：`(B*M, N, 512) -> (B, M*N, 512)`
- 也就是把多个 view 的 tokens **拼到同一个长序列里做 joint attention / joint mamba**

> 这一步是你后续加 “style 分支” 最容易插入的位置之一（我后面会给建议）。

------

### (C) 主干 Encoder：`enc_blocks: 16 blocks`

> 你这份结构是 **14 个 Mamba2Block + 2 个 Transformer Block**（第 8、16 层）

------

## (C1) Mamba2Block（出现在 enc_blocks.0~6, 8~14）

### 输入输出 shape

- 输入：`x ∈ (B, L, 512)`
- 输出：`(B, L, 512)`（残差）

### 结构

1. `norm: RMSNorm(512)`
2. `mamba: Mamba2MultiScan(scan_type="bi")`
   - `in_proj: Linear(512 → 2576)`（**权重形状：2576×512**）
     - 这里 `2576 = 2*d_inner + 2*ngroups*d_state + nheads`
     - 默认：`expand=2 → d_inner=1024`
     - `d_state=256, ngroups=1`
     - `headdim=64 → nheads = 1024/64 = 16`
     - 所以：`2*1024 + 2*1*256 + 16 = 2576`
   - `mamba_scans: 2 个`（双向）
     - 每个 scan 内部都有：
       - `conv1d`: depthwise Conv1d（groups=conv_dim）
         - `conv_dim = d_inner + 2*ngroups*d_state = 1024 + 512 = 1536`
         - `kernel_size = d_conv = 4`
         - 权重形状约：`(1536, 1, 4)`，bias `(1536,)`
       - `dt_bias`: `(16,)`
       - `A_log`: `(16,)`
       - `D`: `(16,)`
       - `norm.weight`: `(1024,)`（RMSNormGated 的权重）
   - `out_proj: Linear(1024 → 512)`（权重形状：512×1024）
3. 残差：`x = x + mamba(norm(x))`

------

## (C2) Transformer Block（出现在 enc_blocks.7 和 enc_blocks.15）

### 输入输出 shape

- 输入：`(B, L, 512)`
- 输出：`(B, L, 512)`（残差）

### 结构（标准 ViT Block）

1. `norm1: LayerNorm(512)`
2. `attn: MemEffAttentionFlash(num_heads=8)`
   - `qkv: Linear(512 → 1536)`（=3*512）
     - **W_qkv shape**：`(1536, 512)`，bias `(1536,)`
   - reshape：
     - `qkv -> (B, L, 3, heads=8, head_dim=64)`
     - `q,k,v -> (B, 8, L, 64)`（内部会做 transpose/permute）
   - attention 输出：
     - `attn_out -> (B, L, 512)`
   - `proj: Linear(512 → 512)`（W: 512×512）
3. 残差：`x = x + attn(norm1(x))`
4. `norm2: LayerNorm(512)`
5. `mlp:`
   - `fc1: Linear(512 → 2048)`（mlp_ratio=4）
   - `GELU`
   - `fc2: Linear(2048 → 512)`
6. 残差：`x = x + mlp(norm2(x))`

------

### (D) `enc_norm: LayerNorm(512)`

- 输出 shape：`(B, L, 512)`

------

### (E) 还原回 3D grid 并用 `deconv` 解码

1. 如果之前做了 multi-view 合并，此处会还原回按 view 的 batch：

- `(B, M*N, 512) -> (B*M, N, 512)`

1. reshape tokens 为 3D grid：

- `(B', N, 512) -> (B', 512, T_lat, H_p, W_p)`
- 例：`(B*M, 512, 16, 44, 80)`

1. `deconv: ConvTranspose3d(512 → output_dims)`

- 默认 `output_dims=12`（你 model.json 里也只看到 deconv，没有 mask/offset 的额外 head key，说明大概率还是 12）
- **关键超参（默认推导）**：
  - `stride = (8,2,2)`
  - `kernel = (9,2,2)`（注意 temporal 多了 1，是为了对齐长度）
  - `padding = (4,0,0)`

#### 输出 shape（核心公式）

- 空间维度：
  - `H_out = (H_p-1)*2 + 2 = 2*H_p = 88 = H_lat`
  - `W_out = 2*W_p = 160 = W_lat`
- 时间维度：
  - `T_out = (T_lat-1)*8 - 2*4 + 9 = 8*T_lat - 7`
  - 如果 `T_lat=16`：`T_out = 8*16 - 7 = 121` ✅

所以 deconv 输出：

- `x ∈ (B', 12, T_out, H_lat, W_lat)`
- 例：`(B*M, 12, 121, 88, 160)`

------

### (F) Gaussian 参数组装（无可学习参数，但结构非常重要）

先把体素/时空网格拉平：

- `(B', 12, T_out, H_lat, W_lat) -> (B', N_g, 12)`
- `N_g = T_out * H_lat * W_lat`

然后拆分 12 维：

- `distance: 1`
- `rgb: 3`
- `scaling: 3`
- `rotation: 4`（四元数）
- `opacity: 1`

并通过 rays 计算 3D 坐标：

- `w = sigmoid(distance + shift)`
- `depth = dnear*(1-w) + dfar*w`
- `pos = rays_o + rays_d * depth`

激活：

- `opacity = sigmoid(opacity - 2)`
- `scale = exp(...)` 再 cap
- `rotation = normalize(q)`
- `rgb = 0.5*tanh(rgb)+0.5`

最终 gaussian 向量：

- `gaussians ∈ (B', N_g, 14)`
   `= [pos(3), opacity(1), scale(3), rotation(4), rgb(3)]`

之后还有：

- multi-view gaussian merge（如果 fuse_multi_views）
- prune（按 opacity）