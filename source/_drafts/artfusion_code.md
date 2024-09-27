---
title: ArtFusion Controllable Arbitrary Style Transfer using Dual Conditional Latent Diffusion Models代码理解
date: 2024-9-19 10:28:00
tags: [风格迁移,diffusion,深度学习,代码]
categories: [风格迁移,diffusion]
comment: false
toc: true
---
#
<!--more-->

# ArtFusion Controllable Arbitrary Style Transfer using Dual Conditional Latent Diffusion Models代码理解

## 1. 环境配置
- 项目地址:https://github.com/ChenDarYen/ArtFusion

    ```bash
    git clone https://github.com/ChenDarYen/ArtFusion.git
    conda env create -f environment.yaml
    conda activate artfusion
    ```

- 下载模型

    - vae: https://ommer-lab.com/files/latent-diffusion/kl-f16.zip
        - 放到`./checkpoints/vae/kl-f16.ckpt`
    - artfusion: https://1drv.ms/u/s!AuZJlZC8oVPfgWC2O77TUlhIfELG?e=RoSa8a
        - 放到`./checkpoints/artfusion/`
    - 注意: artfusion下载过程容易中断,导致下载下来的模型大小不是3G, 注意检查

- 运行代码

    1. 运行`notebooks/style_transfer.ipynb`

        - 如果出现`numpy.core.multiarray failed to import`, 可能是下载的numpy版本不对(不知道为啥会下错), 重新安装:
            ```bash
            pip uninstall numpy
            conda install numpy=1.23.4 --override-channels -c defaults -c pytorch
            pip install numpy==1.23.4
            ```


## 2. 代码结构

- `notebooks/style_transfer.ipynb`: 推断

- `main.py` : 训练

## 3. 代码理解

### 3.1 推断部分(notebooks/style_transfer.ipynb)
- ####  block 1 ： 设置参数，加载模型
```python
# 1. 参数设置
CFG_PATH = '../configs/kl16_content12.yaml' # 配置文件
CKPT_PATH = '../checkpoints/artfusion/artfusion_r12_step=317673.ckpt' # 模型路径
H = 256
W = 256
DDIM_STEPS = 250
ETA = 1.
SEED = 2023
DEVICE = 'cuda'


# 2. 加载模型
import sys
sys.path.append('../')
import os
import random
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning import seed_everything
from PIL import ImageDraw, ImageFont, Image
import matplotlib.pyplot as plt
from einops import rearrange
from omegaconf import OmegaConf
import albumentations
from main import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
# 为了保证实验的「可复现性」，许多机器学习的代码都会有一个方法叫 seed_everything，这个方法尝试固定随机种子以让一些随机的过程在每一次的运行中产生相同的结果。
seed_everything(SEED)

config = OmegaConf.load(CFG_PATH) # 加载配置文件
config.model.params.ckpt_path = CKPT_PATH # 模型路径
config.model.params.first_stage_config.params.ckpt_path = None
model = instantiate_from_config(config.model) # 实例化模型
model = model.eval().to(DEVICE) # 模型加载到设备上, 并设置为eval模式
```


<details style="margin-left: 20px;"><summary>
跳转: instantiate_from_config:  &#9660 ▼>>>
</summary>

```python
def instantiate_from_config(config): #传入的是config.model
    if not "target" in config: # 检查配置文件中是否有target字段
        raise KeyError("Expected key `target` to instantiate.")
    # 第一个括号是返回这个类, 第二个括号是传入这个类的参数将其实例化
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def get_obj_from_str(string, reload=False): # 传入的是config.model.target
    module, cls = string.rsplit(".", 1) # 从右边开始分割, 分割一次
    # module = ldm.models.diffusion.dual_cond_ddpm
    # cls = DualCondDDPM
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    # 动态获取对象的属性值
    # importlib.import_module() 用于动态导入模块, 相当于 import 语句的动态版本
    # getattr() 函数用于返回一个对象属性值
    # 这里将ldm.models.diffusion.dual_cond_ddpm模块导入, 并获取这个文件的一个属性: DualCondDDPM类
    # 为什么要这么麻烦?
    return getattr(importlib.import_module(module, package=None), cls)
```

</details>



- #### block 2 : 图像处理函数
```
定义了三个图像处理函数, 后面再看.
```
- #### block 3 ： 风格图像路径
```python
style_image_paths = [ #风格图像路径
    '../data/styles/d523d66a2f745aff1d3db21be993093fc.jpg',
    '../data/styles/the_scream.jpg',
    '../data/styles/Claude_Monet_73.jpg',
    '../data/styles/430f12a69a198bf3228f8177ed436624c.jpg'
]
# 将多张风格图做stack: [c,h,w] -> [n,c,h,w]
style_images = torch.stack([preprocess_image(p) for p in style_image_paths], dim=0).to(DEVICE)
```

<details style="margin-left: 20px;"><summary>跳转: preprocess_image:  &#9660 ▼>>></summary>

```python
# 3. 图片处理

# 返回tensor图片
def preprocess_image(image_path, size=(W, H)):
    image = Image.open(image_path)
    if not image.mode == "RGB": # 如果不是RGB模式, 转换为RGB
        image = image.convert("RGB")
    image = image.resize(size) # 调整大小
    image = np.array(image).astype(np.uint8) # 转换为numpy数组, 并转换为uint8类型
    image = (image/127.5 - 1.0).astype(np.float32) # 归一化: [0,255]->[0,2]->[-1,1]
    image = rearrange(image, 'h w c -> c h w') #重新排列多维数组的维度
    return torch.from_numpy(image) # 转换为tensor
```
</details>

- #### block 4 ： 显示多张图片
```python
# 显示多张图片
# 输入: tensor图片, 图片张数
display_samples(tensor_to_rgb(style_images), n_columns=len(style_images))
```
<details style="margin-left: 20px;"><summary>跳转: display_samples:  &#9660 ▼>>></summary>


```python
# 
def display_samples(samples, n_columns=1, figsize=(12, 12)):
    # 如果samples是list或tuple, 则将其拼接, 为啥不用stack?
    if isinstance(samples, (list, tuple)):
        samples = torch.cat(samples, dim=0) #samples是一个4维的tensor, [n,c,h,w]
        
    # 将samples排成n_columns列, m行, 然后放到cpu, 转换为numpy, 乘以255
    samples = rearrange(samples, '(n m) c h w -> (m h) (n w) c', n=n_columns).cpu().numpy() * 255.
    # 将numpy数组转换为Image对象
    samples = Image.fromarray(samples.astype(np.uint8))
    plt.rcParams["figure.figsize"] = figsize # 设置图片大小
    plt.imshow(samples) # 显示图片
    plt.axis('off') # 关闭坐标轴
    plt.show()

def tensor_to_rgb(x): #将[-1,1]的tensor转换为[0,1]的rgb图片
    # clamp: 将输入input张量每个元素的夹紧到区间 [min, max]
    return torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
```
</details>

- #### block 5 ： 查看模型学习到的风格

    - 将内容设为0

```python
# 4. 查看模型学习到的风格
bs = len(style_images) # batch size
# ema_scope 是一个上下文管理器，用来应用模型的 EMA（指数移动平均）权重。
# EMA 是一种技术，用于在训练过程中维护模型参数的平滑版本（即对模型参数的滑动平均），通常可以提高生成模型的稳定性和质量。
# "Plotting" 是一个标识符，可能用于标记此上下文管理器的具体用途（这里用于绘图或推理）。
with torch.no_grad(), model.ema_scope("Plotting"): # 关闭梯度, 并使用ema_scope
    # 初始化内容图像[bs,c,h,w]
    c_content = torch.zeros((bs, model.channels, model.image_size, model.image_size)).to(DEVICE)

    # 获取vgg特征
    # model通过参数配置文件可知： ldm.models.diffusion.dual_cond_ddpm.DualCondLDM
    # scaling_layer是一个标准化层，用于对输入数据进行标准化处理: (inp - self.shift) / self.scale
    # model.vgg提取出5个特征层的特征
    vgg_features = model.vgg(model.vgg_scaling_layer(style_images))
    # 将5个vgg风格矩阵的后两个维度（空间维度）的均值方差矩阵拼接在一起
    c_style = model.get_style_features(vgg_features)

    # c1是内容图[4,16,16,16]，c2是风格特征[4,2944]
    c = {'c1': c_content, 'c2': c_style}

    # (x_0, 中间结果)[0] : [batch, 16, 16, 16]
    samples = model.sample_log(cond=c, batch_size=bs, ddim=True, ddim_steps=DDIM_STEPS, eta=1.)[0]

    # [4,3,256,256]
    x_samples = model.decode_first_stage(samples)
    x_samples = tensor_to_rgb(x_samples)

display_samples(x_samples, n_columns=bs)
```

<details style="margin-left: 20px;"><summary>跳转: DualCondLDM.vgg_scaling_layer:  &#9660 ▼>>></summary>

```python
class DualCondLDM(LatentDiffusion): #继承自LatentDiffusion
    def __init__(self, #在前面instantiate_from_config使用
                 style_dim, #配置文件2944
                 cond_stage_key='content_and_std_mean',
                 style_flag_key='style_flag', #styleflag是风格标志？
                 content_flag_key='content_flag',
                 ckpt_path=None, #最开始定义的CKPT_PATH
                 ignore_keys=[], #?
                 load_only_unet=False,
                 scale_factors=None, #list
                 shift_values=None, #list
                 *args, **kwargs):
        super().__init__(model_class=DualConditionDiffusionWrapper, #执行父类的初始化
                         cond_stage_key=cond_stage_key,
                         ckpt_path=None,
                         *args, **kwargs)
        # cond_stage_key用于指定条件的阶段，这里是content_and_std_mean？
        assert cond_stage_key == 'content_and_std_mean'

        self.style_flag_key = style_flag_key
        self.content_flag_key = content_flag_key

        delattr(self, 'scale_factor') #删除父类属性
        # 这里的scale和shift是用来做什么的: 用于对特征进行缩放和平移, 对应adaLn的scale和shift???
        if scale_factors is None: #检查传入的变量是否为None
            #注册缓冲区，scale_factors是一个tensor，单个数值
            self.register_buffer('scale_factors', torch.ones(1))
        else:
            #配置文件中：    
            #scale_factors: [ 0.3335, 0.1840, 0.3386, 0.3695, 0.3052, 0.3254, 0.3262, 0.2794,
            #               0.3670, 0.3812, 0.2283, 0.3122, 0.3555, 0.3291, 0.3485, 0.3699 ]
            self.register_buffer('scale_factors', torch.tensor(scale_factors, dtype=torch.float32))
        if shift_values is None:
            self.register_buffer('shift_values', torch.zeros(1))
        else:
            #配置文件中：
            #shift_values: [ -0.7449, -0.4035,  0.4347, -0.2002, -0.4501,  0.4839, -1.0560,  1.5971,
            #               0.4377, -1.4263, -0.3681, -1.1490,  0.1817, -0.2732, -1.2297, -0.3025 ]
            self.register_buffer('shift_values', torch.tensor(shift_values, dtype=torch.float32))

        #注册一个embedding层，输入为1，输出为style_dim
        self.null_style_vector = torch.nn.Embedding(1, style_dim)
        # 将null_style_vector层的权重初始化为正态分布
        torch.nn.init.normal_(self.null_style_vector.weight, std=0.02)
        if self.use_ema:
            # LitEma是一个EMA的类，用于维护模型参数的平滑版本， 
            # 在ldm.models.diffusion.dual_cond_ddpm中:from ldm.modules.ema import LitEma
            # 获得一个新的层，用于EMA
            self.null_style_vector_ema = LitEma(self.null_style_vector)
            # 打印EMA的数量？
            print(f"Keeping EMAs of {len(list(self.null_style_vector_ema.buffers()))}.")

        # 在ldm.models.diffusion.dual_cond_ddpm中:from ldm.modules.losses.lpips import vgg16, ScalingLayer
        self.vgg = vgg16(pretrained=True, requires_grad=False)
        self.vgg_scaling_layer = ScalingLayer()

        if ckpt_path is not None:
            # 从ckpt_path中初始化模型
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet)

    def get_style_features(self, features, flag=None): #flag是否使用空风格向量
        # features.shape: namedtuple，包含了5个特征层的特征
        # 对每个特征层f, std_mean计算最后两个维度的均值标准差（空间维度（通常是高度和宽度））这样可以得到每个特征图的统计信息，而不考虑特征图的具体空间位置，返回的是一个元组，out[0].shape = (bs, c)
        # 然后cat将每个f均值标准差（bs, c) 在dim=1处，拼接为一个tensor(bs, 2*c)
        # 然后cat将5个f的拼接矩阵在dim=1处，拼接
        # [4, 2*64] [4, 2*128] [4, 2*256] 2*[4, 2*512] -> [4, 2944]
        style_features = torch.cat([torch.cat(torch.std_mean(f, dim=[-1, -2]), dim=1) for f in features], dim=1)
        if flag is not None:
            flag = flag[..., None]
            style_features = torch.where(flag, style_features, self.null_style_vector.weight[0])  # null style
        return style_features

```

</details>


<details style="margin-left: 20px;"><summary>跳转DualCondLDM父类: ldm.modules.diffusion.ddpm.LatenDiffusion:  &#9660 ▼>>></summary>

```python
class LatentDiffusion(pl.LightningModule):
    # classic DDPM with Gaussian diffusion, in image space
    def __init__(self,
                 unet_config,#target: ldm.modules.diffusionmodules.model.StyleUNetModel
                 first_stage_config, #通常用于处理输入数据。target: ldm.models.autoencoder.AutoencoderKL
                 cond_stage_config,#条件阶段的模型配置，控制如何处理条件信息。__is_adaptive__
                 num_timesteps_cond=None, #？
                 cond_stage_key="image",#条件阶段的关键字，指示如何处理条件输入content_and_std_mean
                 cond_stage_trainable=False, #条件阶段是否可训练
                 concat_mode=True, #是否使用拼接模式来处理输入和条件
                 cond_stage_forward=None,
                 conditioning_key=None,#other, other
                 scale_factor=1.0, #特征缩放的因子
                 scale_by_std=False, #是否通过标准差来缩放
                 timesteps=1000,
                 beta_schedule="linear",
                 loss_type="l2",
                 ckpt_path=None,
                 ignore_keys=[], #在加载模型时需要忽略的键
                 load_only_unet=False, #仅加载 U-Net 部分
                 monitor="val/loss", #用于监控验证损失的键，默认为 "val/loss"。？
                 monitor_mode='min', #监控模式，默认为 'min'，表示监控损失最小化
                 use_ema=True, #使用指数移动平均（EMA）
                 first_stage_key="image",
                 image_size=256,
                 channels=3,
                 log_every_t=200, #每多少个时间步记录一次日志
                 clip_denoised=True, #？
                 linear_start=1e-4,
                 linear_end=2e-2,
                 cosine_s=8e-3,
                 given_betas=None,
                 original_elbo_weight=0., #？
                 v_posterior=0.,  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
                 l_simple_weight=1.,
                 # 预测方式：x0 表示预测 x0，eps 表示预测 噪声
                 parameterization="eps",  # all assuming fixed variance schedules
                 scheduler_config=None,
                 use_positional_encodings=False,
                 learn_logvar=False, #是否学习对数方差
                 logvar_init=0.,
                 model_class=DiffusionWrapper, # ?
                 ):
        super().__init__()
        # 如果没有指定条件时间步，则默认为 1
        self.num_timesteps_cond = default(num_timesteps_cond, 1) 
        assert self.num_timesteps_cond <= timesteps #条件时间步小于总时间步
        # for backwards compatibility after implementation of DiffusionWrapper
        if conditioning_key is None:
            # 根据concat_mode的值来选择conditioning_key
            conditioning_key = 'concat' if concat_mode else 'crossattn'
        if cond_stage_config == '__is_unconditional__': #无条件
            conditioning_key = None

        assert parameterization in ["eps", "x0"], 'currently only supporting "eps" and "x0"'
        self.parameterization = parameterization
        print(f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode")
        self.cond_stage_model = None
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.first_stage_key = first_stage_key
        self.image_size = image_size  # try conv?
        self.channels = channels
        self.use_positional_encodings = use_positional_encodings
        # 实例化模型：DualConditionDiffusionWrapper， 传入参数unet_config： ldm.modules.diffusionmodules.model.StyleUNetModel
        self.model = model_class(unet_config, conditioning_key)
        count_params(self.model, verbose=True) #统计模型参数
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler: #使用调度器
            self.scheduler_config = scheduler_config

        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight

        # 注册调度器
        self.register_schedule(given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)

        self.loss_type = loss_type

        self.learn_logvar = learn_logvar
        # 初始化对数方差, 默认不使用学习对数方差，初始化为num_timesteps个logvar_init
        logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
        if self.learn_logvar:
            self.logvar = nn.Parameter(logvar, requires_grad=True)
        else:
            self.register_buffer('logvar', logvar)

        self.scale_by_std = scale_by_std
        self.register_buffer('scale_factor', torch.tensor(scale_factor))

        if monitor is not None:
            self.monitor = monitor
            self.monitor_mode = monitor_mode

        self.concat_mode = concat_mode
        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key
        try:
            self.num_downs = len(first_stage_config.params.ddconfig.ch_mult) - 1
        except:
            self.num_downs = 0
        self.instantiate_first_stage(first_stage_config)
        self.instantiate_cond_stage(cond_stage_config) #cond_stage_model=None
        self.cond_stage_forward = cond_stage_forward #None
        self.clip_denoised = False
        self.bbox_tokenizer = None

        self.restarted_from_ckpt = False #是否从检查点重新启动
        if ckpt_path is not None: #None
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet)
    def register_schedule(self,
                          given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        if exists(given_betas):
            betas = given_betas
        else:
            # shape: [1000]
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        alphas = 1. - betas
        # nm.cumprod函数： 返回给定轴上的元素的累积乘积[a1, a2, a3, ...] -> [a1, a1*a2, a1*a2*a3, ...]
        # alpha_t
        alphas_cumprod = np.cumprod(alphas, axis=0)
        # alpha_t-1
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape #1000
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        # 将numpy数组转换为tensor
        # partial函数：固定函数的部分参数，返回一个新的函数
        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        # （1-v)*beta*(1-alpha_comprod_{t-1})/(1-alpha_comprod_t) + v*beta
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        # log(posterior_variance) , 逐个元素与1e-20取max
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        # beta_t * sqrt(alpha_bar_{t-1}) / (1 - alpha_bar_t)
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        # (1 - alpha_bar_{t-1}) * sqrt(alpha_t) / (1 - alpha_bar_t)
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        # 预测噪声
        if self.parameterization == "eps":
            # beta_t^2 / (2 * p_v * alpha_t * (1 - alpha_cumprod_t))
            lvlb_weights = self.betas ** 2 / (
                    2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            # ?????
            # 0.5 * sqrt(alpha_conprod_t) / (2 * (1 - alpha_comprod_t))
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        else:
            raise NotImplementedError("mu not supported")
        # TODO how to choose this term
        # 将lvlb_weights[0]赋值为lvlb_weights[1]
        # lvlb_weights.shape: [1000]
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        # 检查是否有nan
        assert not torch.isnan(self.lvlb_weights).all()
        # 是否缩短条件时间表
        self.shorten_cond_schedule = self.num_timesteps_cond > 1
        if self.shorten_cond_schedule: #缩短条件时间表, 没走这个分支
            self.make_cond_schedule()
    def make_cond_schedule(self, ):
    '''生成一个条件调度的 ID 列表，初始化时将所有 ID 设置为最后一个时间步，然后将前一部分更新为线性间隔生成的条件 ID。这通常用于控制生成过程中的条件信息。'''
        # 创建一个张量， size:num_timesteps, 用值为num_timesteps-1填充
        # 初始化条件 ID 列表，默认情况下所有 ID 都设为最后一个时间步
        self.cond_ids = torch.full(size=(self.num_timesteps,), fill_value=self.num_timesteps - 1, dtype=torch.long)
        # 创建张量， 从0到num_timesteps-1, 一共num_timesteps_cond个值，四舍五入
        ids = torch.round(torch.linspace(0, self.num_timesteps - 1, self.num_timesteps_cond)).long()
        # 将前num_timesteps_cond 个cond_ids更新为刚刚生成的 ids
        self.cond_ids[:self.num_timesteps_cond] = ids
    def instantiate_first_stage(self, config):
        # 实例化第一阶段模型AutoencoderKL
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train #这里给的空函数
        # 将第一阶段模型的参数设置为不可训练
        for param in self.first_stage_model.parameters():
            param.requires_grad = False
    def instantiate_cond_stage(self, config):
        if not self.cond_stage_trainable:
            if config == "__is_first_stage__": #条件阶段模型 = 第一阶段模型
                print("Using first stage also as cond stage.")
                self.cond_stage_model = self.first_stage_model
            elif config == '__is_adaptive__': #自适应条件模型， 条件阶段模型 = None
                print(f"Training {self.__class__.__name__} as an adaptive conditional model.")
                self.cond_stage_model = None
            elif config == "__is_unconditional__":
                print(f"Training {self.__class__.__name__} as an unconditional model.")
                self.cond_stage_model = None
                # self.be_unconditional = True
            else:
                model = instantiate_from_config(config)
                self.cond_stage_model = model.eval()
                self.cond_stage_model.train = disabled_train
                for param in self.cond_stage_model.parameters():
                    param.requires_grad = False
        else:
            assert config != '__is_first_stage__'
            assert config != '__is_unconditional__'
            model = instantiate_from_config(config)
            self.cond_stage_model = model
    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        # 初始化ldm时没有运行该函数， 在DualCondLDM中运行了
        sd = torch.load(path, map_location="cpu")

        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        # 如果某个键以 ignore_keys 中的某个值开头，则将其从字典中删除。这通常用于忽略一些不需要的参数（例如，不同版本之间的变化）。
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        # 如果only_model为True，则只加载模型参数，否则加载整个状态字典
        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(
            sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")
```

</details>

<details style="margin-left: 20px;"><summary>跳转: make_beta_schedule:  &#9660 ▼>>></summary>

```python
def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == "linear":
        betas = ( # shape: [1000]
                torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
        )

    elif schedule == "cosine":
        timesteps = (
                torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)

    elif schedule == "sqrt_linear":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
    elif schedule == "sqrt":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64) ** 0.5
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas.numpy()
```
</details>

<details style="margin-left: 20px;"><summary>跳转:model_class=ldm.models.diffusion.dual_cond_ddpm.DualConditionDiffusionWrapper &#9660 ▼>>></summary>

```python
class DualConditionDiffusionWrapper(pl.LightningModule):
    def __init__(self, diff_model_config, conditioning_key=('other', 'other')):
        super().__init__()
        # 一个元组，指示如何处理输入的条件。它的长度应该为 2，代表两个条件。
        assert len(conditioning_key) == 2
        self.diffusion_model = instantiate_from_config(diff_model_config)
        self.first_conditioning_key, self.second_conditioning_key = conditioning_key
        assert self.first_conditioning_key in ['concat', 'other'] and \
               self.second_conditioning_key in ['concat', 'other']

    def forward(self,
                x, t, c1: list = None, c2: list = None):
        if self.first_conditioning_key == 'concat': #第一个条件的关键词
            x = torch.cat([x, c1], dim=1) # 将c1连接到输入x的通道维度（dim=1)
            c1 = None

        if self.second_conditioning_key == 'concat':
            x = torch.cat([x, c2], dim=1)
            c2 = None

        return self.diffusion_model(x, t, c1, c2) #调用扩散模型

```

</details>

<details style="margin-left: 20px;"><summary>跳转: self.diffusion_model = instantiate_from_config -> ldm.modules.diffusionmodules.model.StyleUNetModel:  &#9660 ▼>>></summary>

```python
# 在unet基础上扩展一些新功能， 特别是与内容相关的处理
# 加了两个模块：content_in 和 content_adaLN_modulation
class StyleUNetModel(UNetModel):
    # content_refined_dim: 经过处理后的内容特征维度
    def __init__(self, in_channels, content_in_dim, content_refined_dim=None, *args, **kwargs):
        super().__init__(in_channels=in_channels+content_refined_dim, *args, **kwargs)
        # 如果content_refined_dim为None，则设置为content_in_dim
        content_refined_dim = content_refined_dim or content_in_dim
        self.content_in_dim = content_in_dim
        self.content_refined_dim = content_refined_dim
        if content_in_dim != content_refined_dim: #输入的内容特征维度不等于处理后的内容特征维度
            self.content_in = nn.Sequential( #内容输入卷积
                nn.Conv2d(content_in_dim, content_refined_dim, 1),
                nn.SiLU(),
                nn.Conv2d(content_refined_dim, content_refined_dim, 1),
            )
            # 用于得到自适应归一化所需的参数，通常是用于调整内容特征的 scale 和 shift。在风格迁移或生成模型中，动态调整特征的尺度和偏移能够帮助模型更好地适应不同的输入样本。
            self.content_adaLN_modulation = nn.Sequential(
                # 这里的输出维度是输入内容特征维度的两倍。这是为了生成两个归一化参数（scale 和 shift）。
                nn.Linear(self.time_embed_dim, content_in_dim * 2),
                nn.SiLU(),
                nn.Linear(content_in_dim * 2, content_in_dim * 2),
            )

            self.initialize_content_weights()

    def initialize_content_weights(self):
        if hasattr(self, 'content_in'): #如果 content_in 模块存在
            # 将 content_adaLN_modulation 的最后一层（第二个线性层）的权重和偏置初始化为零。这可以帮助稳定训练过程，防止一开始的输出过于偏离零。
            nn.init.constant_(self.content_adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.content_adaLN_modulation[-1].bias, 0)

```

</details>

<details style="margin-left: 20px;"><summary>跳转: ldm.modules.ema.LitEma:  &#9660 ▼>>></summary>

```python
import torch
from torch import nn


class LitEma(nn.Module): #继承自传入的nn.Module
    def __init__(self, model, decay=0.9999, use_num_upates=True):
        super().__init__()
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')

        self.m_name2s_name = {}#用于存储模型参数名称到缓冲区名称的映射
        self.register_buffer('decay', torch.tensor(decay, dtype=torch.float32))
        self.register_buffer('num_updates', torch.tensor(0,dtype=torch.int) if use_num_upates
                             else torch.tensor(-1,dtype=torch.int))

        # 注册模型参数
        for name, p in model.named_parameters():# 遍历模型参数,名称
            if p.requires_grad: #训练时
                #remove as '.'-character is not allowed in buffers
                s_name = name.replace('.','') #去掉参数名称中的点，因为点在 PyTorch 中不允许用作缓冲区名称。
                # 添加新参数名称 到 参数值映射
                self.m_name2s_name.update({name:s_name})
                # 将参数值添加到缓冲区
                self.register_buffer(s_name,p.clone().detach().data)

        self.collected_params = []

    def forward(self,model):
        decay = self.decay

        if self.num_updates >= 0:
            self.num_updates += 1
            decay = min(self.decay,(1 + self.num_updates) / (10 + self.num_updates))

        one_minus_decay = 1.0 - decay

        with torch.no_grad():
            m_param = dict(model.named_parameters())
            shadow_params = dict(self.named_buffers())

            for key in m_param:
                if m_param[key].requires_grad:
                    sname = self.m_name2s_name[key]
                    shadow_params[sname] = shadow_params[sname].type_as(m_param[key])
                    shadow_params[sname].sub_(one_minus_decay * (shadow_params[sname] - m_param[key]))
                else:
                    assert not key in self.m_name2s_name

    def copy_to(self, model):
        m_param = dict(model.named_parameters())
        shadow_params = dict(self.named_buffers())
        for key in m_param:
            if m_param[key].requires_grad:
                m_param[key].data.copy_(shadow_params[self.m_name2s_name[key]].data)
            else:
                assert not key in self.m_name2s_name

    def store(self, parameters):
        """
        Save the current parameters for restoring later.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            temporarily stored.
        """
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored parameters.
        """
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

```

</details>
<details style="margin-left: 20px;"><summary>跳转: ldm.modules.losses.lpips.vgg16:  &#9660 ▼>>></summary>

```python
import torch
import torch.nn as nn
from torchvision import models
from collections import namedtuple

from ldm.util import get_ckpt_path
class vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        # 从torchvision.models中加载预训练的vgg16模型
        vgg_pretrained_features = models.vgg16(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            # 将vgg16模型的前4层添加到slice1中
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out #out的类型是namedtuple，包含了5个特征层的特征

```

</details>
<details style="margin-left: 20px;"><summary>跳转: ldm.modules.losses.lpips.ScalingLayer:  &#9660 ▼>>></summary>

```python
#对输入进行标准化处理
class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])
    #将输入数据的值缩放到更合适的范围，以提高模型训练的稳定性和效果。
    def forward(self, inp):
        return (inp - self.shift) / self.scale

```

</details>

<details style="margin-left: 20px;"><summary>跳转: samples = model.sample_log -> ldm.models.diffusion.ddpm.LatentDiffusion.sample_log:  &#9660 ▼>>></summary>

```python
class LatentDiffusion(pl.LightningModule):
    @torch.no_grad()
    def sample_log(self,cond,batch_size,ddim, ddim_steps,**kwargs):

        if ddim:
            ddim_sampler = DDIMSampler(self)
            # 16,16,16(特征通道，所以16？)
            shape = (self.channels, self.image_size, self.image_size)
            # x_0, 中间结果
            samples, intermediates = ddim_sampler.sample(ddim_steps,batch_size, shape,cond,verbose=False,**kwargs)

        else:
            samples, intermediates = self.sample(cond=cond, batch_size=batch_size, return_intermediates=True,**kwargs)

        return samples, intermediates
```
</details>

<details style="margin-left: 20px;"><summary>跳转: ddim_sampler = DDIMSampler -> ldm.models.diffusion.ddim.DDIMSampler:  &#9660 ▼>>></summary>

```python
class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model #DualCondLDM
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule
```

</details>

<details style="margin-left: 20px;"><summary>跳转: samples, intermediates = ddim_sampler.sample -> ldm.models.diffusion.ddim.DDIMSampler.sample:  &#9660 ▼>>></summary>

```python
class DDIMSampler(object):
    @torch.no_grad()
    def sample(self,
               S, #ddim采样的时间步数
               batch_size,
               shape,
               conditioning=None, #传入的条件c{c1:, c2:}
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               unconditional_guidance_scale_2=1.,
               unconditional_conditioning_2=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                # cbs: conditionings batch size
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size: #判断风格图像数量是否与batch_size相等
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    unconditional_guidance_scale_2=unconditional_guidance_scale_2,
                                                    unconditional_conditioning_2=unconditional_conditioning_2,
                                                    )
        return samples, intermediates。请说中文
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout, #0
                                                    temperature=temperature, #1
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t, #100 log：记录
                                                    unconditional_guidance_scale=unconditional_guidance_scale, #1
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    unconditional_guidance_scale_2=unconditional_guidance_scale_2, #1
                                                    unconditional_conditioning_2=unconditional_conditioning_2,
                                                    )
        return samples, intermediates #x_0 , 中间结果
    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        # 对输入的x： 复制，分离，转换为torch.float32类型，转移到模型设备
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        # 干嘛不直接用self.model.alphas_cumprod： 
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        # 直接copy过来不行嘛？
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100, #log: 记录
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                      unconditional_guidance_scale_2=1., unconditional_conditioning_2=None,):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None: #走这个分支
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]} #中间结果
        #np.flip(timesteps)：翻转timesteps
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        # 250
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            # ts:[step, step, step, step]batch个step
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            '''!!!使用遮罩允许在采样过程中对图像的特定区域进行控制。例如，可以只对某些区域应用去噪，而保留其他区域的原始信息。???'''
            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img

            # x_t -> x_{t-1}
            outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning,
                                      unconditional_guidance_scale_2=unconditional_guidance_scale_2,
                                      unconditional_conditioning_2=unconditional_conditioning_2,
                                      )
            img, pred_x0 = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                      unconditional_guidance_scale_2=1., unconditional_conditioning_2=None):
    # t:DDPM的原始时间步数， index: 当前t是ddim中的第几个时间步
        b, *_, device = *x.shape, x.device

        x_in = x
        t_in = t
        c_in = copy.deepcopy(c) if isinstance(c, dict) else c
        uncond_guidance = unconditional_conditioning is not None and unconditional_guidance_scale != 1. # false
        uncond_guidance_2 = unconditional_conditioning_2 is not None and unconditional_guidance_scale_2 != 1. # false
        if uncond_guidance: #false
            x_in = torch.cat([x_in, x])
            t_in = torch.cat([t_in, t])
            if isinstance(c_in, dict):
                for key in c_in.keys():
                    c_in[key] = torch.cat([c_in[key], unconditional_conditioning[key]])
            else:
                c_in = torch.cat([c_in, unconditional_conditioning])
        if uncond_guidance_2: #false
            x_in = torch.cat([x_in, x])
            t_in = torch.cat([t_in, t])
            if isinstance(c_in, dict):
                for key in c_in.keys():
                    c_in[key] = torch.cat([c_in[key], unconditional_conditioning_2[key]])
            else:
                c_in = torch.cat([c_in, unconditional_conditioning_2])

        # 使用DualCondLDM的预测出噪声
        e_t = self.model.apply_model(x_in, t_in, c_in)

        # 这一部分也没走
        if uncond_guidance and not uncond_guidance_2:
            e_t, e_t_uncond = e_t.chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
        elif not uncond_guidance and uncond_guidance_2:
            e_t, e_t_uncond = e_t.chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale_2 * (e_t - e_t_uncond)
        elif uncond_guidance and uncond_guidance_2:
            e_t, e_t_uncond, e_t_uncond_2 = e_t.chunk(3)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond) + \
                  e_t_uncond_2 + unconditional_guidance_scale_2 * (e_t - e_t_uncond_2) - e_t

        if score_corrector is not None: #false
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        # a_t [b, 1, 1, 1]
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        # a_{t-1} [b, 1, 1, 1]
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        # sigma_t [b, 1, 1, 1]
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised: #false
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        # [batch, 16, 16, 16]
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0 #x_t-1 , x_0^hat
```
</details>

<details style="margin-left: 20px;"><summary>跳转: make_ddim_timesteps -> ldm.modules.diffusionmodules.util.make_ddim_timesteps:  &#9660 ▼>>></summary>

```python
def make_ddim_timesteps(ddim_discr_method, num_ddim_timesteps, num_ddpm_timesteps, verbose=True):
    if ddim_discr_method == 'uniform': #离散化方法为均匀
        # c为ddim的采样步长
        c = num_ddpm_timesteps // num_ddim_timesteps
        ddim_timesteps = np.asarray(list(range(0, num_ddpm_timesteps, c)))
    elif ddim_discr_method == 'quad': #二次离散化
        #[0, ... , sqrt(0.8*num_ddpm_timesteps)]^2 step=ddim采样步数
        ddim_timesteps = ((np.linspace(0, np.sqrt(num_ddpm_timesteps * .8), num_ddim_timesteps)) ** 2).astype(int)
    else:
        raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')

    # assert ddim_timesteps.shape[0] == num_ddim_timesteps
    # add one to get the final alpha values right (the ones from first scale to data during sampling)
    # 对生成的时间步加1，以确保最终的α值正确。这是因为在采样过程中，通常需要从0开始到1进行缩放。
    steps_out = ddim_timesteps + 1
    if verbose: #打印ddim采样步数
        print(f'Selected timesteps for ddim sampler: {steps_out}')
    return steps_out
```
</details>

<details style="margin-left: 20px;"><summary>跳转: make_ddim_sampling_parameters -> ldm.modules.diffusionmodules.util:  &#9660 ▼>>></summary>

```python
def make_ddim_sampling_parameters(alphacums, ddim_timesteps, eta, verbose=True):
    # select alphas for computing the variance schedule
    # 选择ddim采样的时间步对应的alpha值
    alphas = alphacums[ddim_timesteps]
    alphas_prev = np.asarray([alphacums[0]] + alphacums[ddim_timesteps[:-1]].tolist())

    # according the the formula provided in https://arxiv.org/abs/2010.02502
    # 重要的是这个sigma
    sigmas = eta * np.sqrt((1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev))
    if verbose:
        print(f'Selected alphas for ddim sampler: a_t: {alphas}; a_(t-1): {alphas_prev}')
        print(f'For the chosen value of eta, which is {eta}, '
              f'this results in the following sigma_t schedule for ddim sampler {sigmas}')
    return sigmas, alphas, alphas_prev
```
</details>

<details style="margin-left: 20px;"><summary>跳转: e_t = self.model.apply_model -> ldm.models.diffusion.dual_cond_ddpm.DualCondLDM:  &#9660 ▼>>></summary>

```python
class DualCondLDM(LatentDiffusion):
    def apply_model(self, x_noisy, t, cond, return_ids=False):
        # DualConditionDiffusionWrapper -> StyleUNetModel -> UNetModel
        # x_noisy 与 cond{c1} concat作为输入
        # t 与 cond{c2} 相加作为emb
        x_recon = self.model(x_noisy, t, **cond)

        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon #[batch, 16, 16, 16]
```
</details>

<details style="margin-left: 20px;"><summary>跳转: x_recon = self.model-> ldm.models.diffusion.dual_cond_ddpm.DualConditionDiffusionWrapper  &#9660 ▼>>></summary>

```python
# 主要用于封装和处理双条件扩散模型
class DualConditionDiffusionWrapper(pl.LightningModule):
    def forward(self,
                x, t, c1: list = None, c2: list = None):
        if self.first_conditioning_key == 'concat': #第一个条件的关键词
            x = torch.cat([x, c1], dim=1) # 将c1连接到输入x的通道维度（dim=1)
            c1 = None

        if self.second_conditioning_key == 'concat':
            x = torch.cat([x, c2], dim=1)
            c2 = None

        return self.diffusion_model(x, t, c1, c2) #调用扩散模型
```
</details>

<details style="margin-left: 20px;"><summary>跳转: return self.diffusion_model -> ldm.modules.diffusionmodules.model.StyleUNetModel  &#9660 ▼>>></summary>

```python
class StyleUNetModel(UNetModel):
    def forward(self, x, timesteps, content, style, *args, **kwargs):
        # content是c1, style是c2
        emb = self.time_embed(timesteps) #[batch, 1] -> [batch, hidden_size]
        if self.content_in_dim != self.content_refined_dim:
            # [batch, time_embed_dim] -> [batch, 2*content_in_dim]
            # 然后从dim=1开始切分为两个张量得到shift和scale[batch, content_in_dim]
            shift, scale = self.content_adaLN_modulation(emb)[..., None, None].chunk(2, dim=1)
            # modulate函数用于调整内容特征的尺度和偏移:content*(1+scale)+shift
            # [batch, content_in_dim, H, W](4,16,16,16) -> [batch, content_refined_dim, H, W] (4,12,16,16)
            content = self.content_in(modulate(content, shift, scale))

        #将调整后的内容特征与输入特征连接
        # [batch, 16, 16, 16] + [batch, 12, 16, 16] -> [batch, 28, 16, 16]
        x = torch.cat((x, content), dim=1)
        return super().forward(x, emb, style)

class UNetModel(nn.Module):
    # x: [batch, 28, 16, 16] ： x + c1_scaled
    # emb: [batch, hidden_size]
    def forward(self, x, emb, context=None, y=None, *args, **kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn or ada
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None and self.use_label_emb
        ), "must specify y if and only if the model is class-conditional"
        hs = []
        # emb = self.time_embed(timesteps)

        # 不使用label
        if self.num_classes is not None and self.use_label_emb:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y, self.training)

        # 这里的context是style, 也就是c2
        if context is not None and not self.use_spatial_transformer:
            # emb = t + c2
            # context_emb与time_emb类似
            # [batch, hidden_size] + [batch, hidden_size] -> [batch, hidden_size]
            emb = emb + self.context_emb(context)
            context = None

        h = x.type(self.dtype)
        for module in self.input_blocks:
            # 这里context是none, 所以还是正常unet输入
            h = module(h, emb, context)
            hs.append(h)

        h = self.middle_block(h, emb, context)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)

        h = h.type(x.dtype)
        if self.predict_codebook_ids:
            return self.id_predictor(h)
        else:
            h = self.out(h, emb)
            return h #[batch, 16, 16, 16]
```
</details>

<details style="margin-left: 20px;"><summary>跳转: emb = self.time_embed -> TimestepEmbedder.forward  &#9660 ▼>>></summary>

```python
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            #emb(512) -> hidden_size(1024)
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            # hidden_size(1024) -> hidden_size(1024)
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod #[batch, 1] -> [batch, embedding_size]
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp( # exp( -log(10000) * [0, 1, 2, ... , half]/half )
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None] # [batch,1] * [1, half] = [batch, half]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1) #[batch, dim]
        if dim % 2: #dim%2 ！=0要多加一列
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb
```
</details>


<details style="margin-left: 20px;"><summary>跳转: x_samples = model.decode_first_stage -> ldm.models.diffusion.dual_cond_ddpm.DualCondLDM  &#9660 ▼>>></summary>

```python
    def decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
        for i in range(z.shape[1]): #对z的每个通道进行缩放
            # scale_factor和shift_value在配置文件中设置， 用于将z的值缩放到合适的范围以供解码器使用
            # 这里[:,i] 是对z的第i个通道进行缩放
            z[:, i] = z[:, i] / self.scale_factors[i] + self.shift_values[i]
        return self.first_stage_model.decode(z) #解码器部分暂时不做解释？？？
```
</details>

<details style="margin-left: 20px;"><summary>跳转: model.ema_scope("Plotting") -> ldm.models.diffusion.dual_cond_ddpm.DualCondLDM  &#9660 ▼>>></summary>

```python
class DualCondLDM(LatentDiffusion):
    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            self.null_style_vector_ema.store(self.null_style_vector.parameters())
            self.null_style_vector_ema.copy_to(self.null_style_vector)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally: #直接到这？？？？
            if self.use_ema:
                # model_ema继承自LatentDiffusion
                self.model_ema.restore(self.model.parameters())
                self.null_style_vector_ema.restore(self.null_style_vector.parameters())
                if context is not None: #输出消息表示已恢复训练权重
                    print(f"{context}: Restored training weights")

```
</details>

<details style="margin-left: 20px;"><summary>跳转: self.model_ema.store -> ldm.modules.ema.LitEma  &#9660 ▼>>></summary>

```python
class LitEma(nn.Module):
    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored parameters.
        """
        '''
        用于恢复之前通过 store 方法保存的参数,
        在不影响原始优化过程的情况下，验证模型的 EMA（指数移动平均）参数
        在进行验证或保存模型之前，先调用 store 方法保存当前参数；在验证后，可以使用 restore 方法恢复原参数
        '''
        # 通过 zip 将 self.collected_params（先前保存的参数）与传入的 parameters（需要更新的参数）配对
        # self.collected_params 是一个包含 EMA 参数的列表或集合，通常在模型训练过程中会定期更新。
        for c_param, param in zip(self.collected_params, parameters):
            # param.data 是 parameters 中每个参数的实际数据
            # c_param.data 是收集到的 EMA 参数的数据
            # copy_ 方法用于将 c_param.data 的值复制到 param.data 中，更新当前参数为之前保存的 EMA 参数的值
            param.data.copy_(c_param.data)
```
</details>

- #### block 6 ： 风格迁移

```python
def get_content_style_features(content_image_path, style_image_path, h=H, w=W):
    style_image = preprocess_image(style_image_path)[None, :].to(DEVICE)
    content_image = preprocess_image(content_image_path, size=(w, h))[None, :].to(DEVICE)
    
    with torch.no_grad(), model.ema_scope("Plotting"):
        vgg_features = model.vgg(model.vgg_scaling_layer(style_image))
        c_style = model.get_style_features(vgg_features)
        null_style = c_style.clone()
        null_style[:] = model.null_style_vector.weight[0]

        content_encoder_posterior = model.encode_first_stage(content_image)
        content_encoder_posterior = model.get_first_stage_encoding(content_encoder_posterior)
        c_content = model.get_content_features(content_encoder_posterior)
        null_content = torch.zeros_like(c_content)
        
    c = {'c1': c_content, 'c2': c_style}
    c_null_style = {'c1': c_content, 'c2': null_style}
    c_null_content = {'c1': null_content, 'c2': c_style}
    
    return c, c_null_style, c_null_content
    

def style_transfer(
    content_image_path, style_image_path,
    h=H, w=W,
    content_s=1., style_s=1.,
    ddim_steps=DDIM_STEPS, eta=ETA,
):
    c, c_null_style, c_null_content = get_content_style_features(content_image_path, style_image_path, h, w)
    
    with torch.no_grad(), model.ema_scope("Plotting"):
        samples = model.sample_log(
            cond=c, batch_size=1, x_T = torch.rand_like(c['c1']),
            ddim=True, ddim_steps=ddim_steps, eta=eta,
            unconditional_guidance_scale=content_s, unconditional_conditioning=c_null_content,
            unconditional_guidance_scale_2=style_s, unconditional_conditioning_2=c_null_style)[0]

        x_samples = model.decode_first_stage(samples)
        x_samples = tensor_to_rgb(x_samples)
    
    return x_samples

content_image_path = '../data/contents/lofoton.jpg'
style_image_path = '../data/styles/d523d66a2f745aff1d3db21be993093fc.jpg'

style_image = preprocess_image(style_image_path)[None, :]
content_image = preprocess_image(content_image_path)[None, :]
display_samples((tensor_to_rgb(content_image), tensor_to_rgb(style_image)), figsize=(6, 3), n_columns=2)
x_samples = style_transfer(content_image_path, style_image_path, content_s=0.5, style_s=2.)
display_samples(x_samples, figsize=(3, 3))
```