---
title: 12.4 锚框
date: 2024-8-12 17:00:00
tags: [机器学习,pytorch]
categories: [机器学习]
comment: true
toc: true
---

#### 

<!--more-->

# 4 锚框

- 目标检测算法通常会在输入图像中采样大量的区域，然后判断这些区域中是否包含我们感兴趣的目标，并调整区域边界从而更准确地预测目标的真实边界框（ground‐truthboundingbox）

- 不同的模型使用的区域采样方法可能不同。这里我们介绍其中的一种方法：以每个像素为中心，生成多个缩放比和宽高比（aspectratio）不同的边界框。这些边界框被称为锚框（anchorbox）

## 4.1 生成多个锚框

- 假设输入图像的高度为h，宽度为w。我们以图像的每个像素为中心生成不同形状的锚框：缩放比为$s \in (0,1]$和宽高比为$r > 0$。那么锚框的宽和高将分别为$ws\sqrt{r}$和$hs/\sqrt{r}$。

- 要生成多个不同形状的锚框，让我们设置许多缩放比（scale）取值$s_1, \ldots, s_n$和多个宽高比$r_1, \ldots, r_m$。如果以每个像素为中心生成所有的锚框，输入图像将一共得到$whnm$个锚框。

- 尽管这些锚框可能会覆盖所有真实边界框，但计算复杂性很容易过高。在实践中，我们只考虑包含$s_1$或$r_1$的组合。

    $$(s_ 1, r_ 1), (s_ 1, r_ 2), \ldots, (s_ 1, r_ m), (s_ 2, r_ 1), (s_ 3, r_ 1), \ldots, (s_ n, r_ 1).$$

    - 也就是说，以同一像素为中心的锚框的数量是n+m−1。对于整个输入图像，将共生成wh(n+m−1)个锚框。


```python
# 修改输出精度，以获得更简洁的输出。
%matplotlib inline
import torch
from d2l import torch as d2l
torch.set_printoptions(2) # 精简输出精度

#@save
def multibox_prior(data, sizes, ratios):
    '''生成以每个像素为中心具有不同形状的锚框'''
    in_height, in_width = data.shape[-2:] #data的形状: batch x channel x height x width
    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
    boxes_per_pixel = (num_sizes + num_ratios - 1) #每个像素的锚框数
    size_tensor = torch.tensor(sizes, device=device)
    ratio_tensor = torch.tensor(ratios, device=device)

    # 为了将锚点移动到像素的中心，需要设置偏移量。
    # 因为一个像素的高为1且宽为1，我们选择偏移0.5。
    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height  # 行步长, 用于在每个像素的中心生成锚框
    steps_w = 1.0 / in_width  # 列步长

    # 生成锚框的所有中心点
    center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h #arange对应每个像素
    center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w
    shift_y, shift_x = torch.meshgrid(center_h, center_w, indexing='ij')
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)

    # 生成“boxes_per_pixel”个高和宽，之后用于创建锚框的四角。
    w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]),
                   sizes[0] * torch.sqrt(ratio_tensor[1:])))\
                   * in_height / in_width  # 相对于坐标的宽
    h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]),
                     sizes[0] / torch.sqrt(ratio_tensor[1:])))  # 相对于坐标的高

    # 除以2来获得半高和半宽
    anchor_manipulations = torch.stack(
        (-w, -h, w, h)).T.repeat(in_height * in_width, 1) / 2

    # 每个中心点都将有“boxes_per_pixel”个锚框，所以生成含所有锚框中心的网格, 重复了“boxes_per_pixel”次
    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y],
                           dim=1).repeat_interleave(boxes_per_pixel, dim=0)
    output = out_grid + anchor_manipulations
    return output.unsqueeze(0) #output的形状: (1, out_height * out_width * boxes_per_pixel, 4)

img = d2l.plt.imread('./img/catdog.jpg')
h, w = img.shape[:2]
print(h, w)
X = torch.rand(size=(1, 3, h, w))  # 构造输入数据
Y = multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
Y.shape #（批量大小，锚框的数量，4）
```

    718 931
    




    torch.Size([1, 3342290, 4])



- 将锚框变量Y的形状更改为(图像高度,图像宽度,以同一像素为中心的锚框的数量,4)后，我们可以获得以指定像素的位置为中心的所有锚框。


```python
boxes = Y.reshape(h, w, 5, 4)
boxes[250, 250, 0, :]
```




    tensor([-0.02, -0.03,  0.56,  0.72])



- 为了显示以图像中以某个像素为中心的所有锚框，定义下面的show_bboxes函数来在图像上绘制多个边界框。


```python
#@save
def show_bboxes(axes, bboxes, labels=None, colors=None):
    '''显示所有边界框'''
    def _make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj

    labels = _make_list(labels)
    colors = _make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = d2l.bbox_to_rect(bbox.detach().numpy(), color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i], va='center',
                      ha='center', fontsize=9, color=text_color,
                      bbox=dict(facecolor=color, lw=0))
            

d2l.set_figsize()
#变量boxes中x轴和y轴的坐标值已分别除以图像的宽度和高度。绘制锚框时，我们需要恢复它们原始的坐标值
bbox_scale = torch.tensor((w, h, w, h)) #
fig = d2l.plt.imshow(img)
show_bboxes(fig.axes, boxes[250, 250, :, :] * bbox_scale,
            ['s=0.75, r=1', 's=0.5, r=1', 's=0.25, r=1', 's=0.75, r=2',
             's=0.75, r=0.5'])
```


    
![svg](img/deeplearning/code/pytorch/12_computer_vision/4_anchor_box_files/4_anchor_box_5_0.svg)
    


## 4.2 交并比(IoU)

- 如果已知目标的真实边界框，那么这里的“好”该如何如何量化呢？直观地说，可以衡量锚框和真实边界框之间的相似性。

- 杰卡德系数（Jaccard）:

    $$ J(A,B) = \frac{ |A \cap B | }{ | A \cup B | }$$

    - 我们可以将任何边界框的像素区域视为一组像素集合

    - 交并比的取值范围在0和1之间：0表示两个边界框无重合像素，1表示两个边界框完全重合。

- 给定两个锚框或边界框的列表，以下box_iou函数将在这两个列表中计算它们成对的交并比


```python
# 给定两个锚框或边界框的列表，以下box_iou函数将在这两个列表中计算它们成对的交并比
#@save
def box_iou(boxes1, boxes2):
    '''计算两个锚框或边界框列表中成对的交并比'''
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *
                              (boxes[:, 3] - boxes[:, 1]))
    # `boxes1`, `boxes2`, `areas1`, `areas2`的形状: 
    # `boxes1`：(boxes1的数量, 4),
    # `boxes2`：(boxes2的数量, 4),
    # `areas1`：(boxes1的数量,),
    # 'areas2`：(boxes2的数量,)
    areas1 = box_area(boxes1) #计算每个锚框的面积
    areas2 = box_area(boxes2)
    # `inter_upperlefts`, `inter_lowerrights`, `inters`的形状:
    # (boxes1的数量, boxes2的数量, 2)
    inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2]) #计算交集的左上角坐标, 这里比较的是两个坐标, 如max ( (x1, y1), (x2, y2) ) = (max(x1, x2), max(y1, y2))
    inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) #计算交集的右下角坐标
    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0) #计算交集的面积
    # `inter_areas` and `union_areas`的形状: (boxes1的数量, boxes2的数量)
    inter_areas = inters[:, :, 0] * inters[:, :, 1] #计算交集的面积
    union_areas = areas1[:, None] + areas2 - inter_areas #计算并集的面积, 这里用none是为了让两个面积相加, 因为boxes1和boxes2的数量不一定相等
    return inter_areas / union_areas
```

## 4.3 在训练数据中标注锚框

- 在训练集中，我们将每个锚框视为一个训练样本。为了训练目标检测模型，我们需要每个锚框的类别（class）和偏移量（offset）标签，其中前者是与锚框相关的对象的类别，后者是真实边界框相对于锚框的偏移量。

- 预测时，我们为每个图像生成多个锚框，预测所有锚框的类别和偏移量，根据预测的偏移量调整它们的位置以获得预测的边界框，最后只输出符合特定条件的预测边界框。

- 目标检测训练集带有真实边界框的位置及其包围物体类别的标签。要标记任何生成的锚框，我们可以参考分配到的最接近此锚框的真实边界框的位置和类别标签。下文将介绍一个算法，它能够把最接近的真实边界框分配给锚框。

    - 给定图像，假设锚框是$A_1, \ldots, A_{ n_ a}$，真实边界框是$B_1, \ldots, B_{ n_ b }$，其中$n_a \geq n_b$。定义矩阵$X \in \mathbb{R}^{n_a \times n_b}$，其中第$i$行第$j$列的元素$x_{ij}$是锚框$A_i$和真实边界框$B_j$的交并比。
    
        1. 在矩阵X中找到最大元素，将其行索引和列索引表示为$i_1$和$j_1$。然后真实边界框$B_{j_1}$被分配给锚框$A_{i_1}$。分配后, 丢弃矩阵X的第$i_1$行和第$j_1$列中的所有元素。

        2. 重复1. 直至丢弃了矩阵$X$中$n_b$列.

        3. 剩下$n_a - n_b$个锚框. 例如给定任何锚框$A_i$，在矩阵$X$的第$i$行中找到与$A_i$的IoU最大的真实边界框$B_j$，只要该IoU大于预先设定的阈值，例如0.5。然后真实边界框$B_j$被分配给锚框$A_i$。


```python
#@save
def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    '''将最接近真实边界框分配给锚框'''
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    # 位于第i行和第j列的元素 x_ij 是锚框 i 和真实边界框 j 的IoU
    jaccard = box_iou(anchors, ground_truth)
    # 对于每个锚框，分配的真实边界框的张量
    anchors_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long,
                                  device=device) # torch.full()返回一个张量，包含了指定标量值的张量
    # 根据阈值，决定是否分配真实边界框
    max_ious, indices = jaccard.max(dim=1) #返回每行的最大值和最大值的索引, dim=1对应每个锚框得到n_a个iou值
    anc_i = torch.nonzero(max_ious >= iou_threshold).reshape(-1) #返回非零元素的索引, 对于这n_a个iou值, 如果大于阈值, 则返回行索引
    box_j = indices[max_ious >= iou_threshold] # 满足要求的值的列索引
    anchors_bbox_map[anc_i] = box_j
    # 为每个真实边界框分配锚框
    col_discard = torch.full((num_anchors,), -1)
    row_discard = torch.full((num_gt_boxes,), -1)
    for _ in range(num_gt_boxes):
        max_idx = torch.argmax(jaccard)
        box_idx = (max_idx % num_gt_boxes).long()
        anc_idx = (max_idx / num_gt_boxes).long()
        anchors_bbox_map[anc_idx] = box_idx
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard
    return anchors_bbox_map
```

### 4.3.1 标记类别和偏移量

- 假设一个锚框A被分配了一个真实边界框B。一方面，锚框A的类别将被标记为与B相同。另一方面，锚框A的偏移量将根据B和A中心坐标的相对位置以及这两个框的相对大小进行标记。

- 鉴于数据集内不同的框的位置和大小不同，我们可以对那些相对位置和大小应用变换，使其获得分布更均匀且易于拟合的偏移量:

    - 给定框A和B，中心坐标分别为$(x_a, y_a)$和$(x_b, y_b)$，宽度和高度分别为$w_a, h_a$和$w_b, h_b$。那么框B相对于框A的偏移量可以表示为:

        $$\left( \frac{ \frac{x_b - x_a}{w_a} - \mu_x }{ \sigma_x }, \frac{ \frac{y_b - y_a}{h_a} - \mu_y }{ \sigma_y }, \frac{ \log( \frac{w_b}{w_a} ) - \mu_w }{ \sigma_w }, \frac{ \log( \frac{h_b}{h_a} ) - \mu_h }{ \sigma_h } \right)$$

        - 其中常量的默认值为$\mu_x = \mu_y = \mu_w = \mu_h = 0, \sigma_x = \sigma_y = 0.1, \sigma_w = \sigma_h = 0.2$。


```python
#@save
def offset_boxes(anchors, assigned_bb, eps=1e-6):
    '''对锚框偏移量的转换'''
    c_anc = d2l.box_corner_to_center(anchors)
    c_assigned_bb = d2l.box_corner_to_center(assigned_bb)
    offset_xy = 10.0 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
    offset_wh = 5.0 * torch.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:])
    offset = torch.cat([offset_xy, offset_wh], dim=1)
    return offset
```

- 如果一个锚框没有被分配真实边界框，我们只需将锚框的类别标记为背景（background）背景类别的锚框通常被称为负类锚框

- 使用真实边界框（labels参数）实现以下multibox_target函数，来标记锚框的类别和偏移量（anchors参数）。此函数将背景类别的索引设置为零，然后将新类别的整数索引递增一。


```python
#@save
def multibox_target(anchors, labels):
    '''使用真实边界框标记锚框'''
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.device, anchors.shape[0]
    for i in range(batch_size):
        label = labels[i, :, :]
        anchors_bbox_map = assign_anchor_to_bbox(label[:, 1:], anchors, device)
        bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(1, 4)
        # 将类标签和偏移量初始化为零
        class_labels = torch.zeros(num_anchors, dtype=torch.long, device=device)
        assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32,
                                  device=device)
        # 使用真实边界框来标记锚框的类别
        # 如果一个锚框没有被分配真实边界框，则类别为背景
        indices_true = torch.nonzero(anchors_bbox_map >= 0)
        bb_idx = anchors_bbox_map[indices_true]
        class_labels[indices_true] = label[bb_idx, 0].long() + 1
        assigned_bb[indices_true] = label[bb_idx, 1:]
        # 偏移量转换
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)
    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    class_labels = torch.stack(batch_class_labels)
    return (bbox_offset, bbox_mask, class_labels)
```

- 下面通过一个具体的例子来说明锚框标签。其中第一个元素是类别（0代表狗，1代表猫），其余四个元素是左上角和右下角的(x,y)轴坐标（范围介于0和1之间）。我们还构建了五个锚框，用左上角和右下角的坐标进行标记：A0,...,A4（索引从0开始）。然后我们在图像中绘制这些真实边界框和锚框。


```python
ground_truth = torch.tensor([[0, 0.1, 0.08, 0.52, 0.92],
                                [1, 0.55, 0.2, 0.9, 0.88]])
anchors = torch.tensor([[0, 0.1, 0.2, 0.3], [0.15, 0.2, 0.4, 0.4],
                            [0.63, 0.05, 0.88, 0.98], [0.66, 0.45, 0.8, 0.8],
                            [0.57, 0.3, 0.92, 0.9]])

fig = d2l.plt.imshow(img)
show_bboxes(fig.axes, ground_truth[:, 1:] * bbox_scale, ['dog', 'cat'], 'k')
show_bboxes(fig.axes, anchors * bbox_scale, ['0', '1', '2', '3', '4'])
```


    
![svg](img/deeplearning/code/pytorch/12_computer_vision/4_anchor_box_files/4_anchor_box_15_0.svg)
    


- 背景、狗和猫的类索引分别为0、1和2。下面我们为锚框和真实边界框样本添加一个维度。


```python
labels = multibox_target(anchors.unsqueeze(dim=0), ground_truth.unsqueeze(dim=0))
```

- 返回的结果中有三个元素，都是张量格式。第三个元素包含标记的输入锚框的类别。

- 分析上面的图片:

    - 在所有的锚框和真实边界框配对中，锚框A4与猫的真实边界框的IoU是最大的。因此，A4的类别被标记为猫。

    - 去除包含A4或猫的真实边界框的配对，在剩下的配对中，锚框A1和狗的真实边界框有最大的IoU。因此，A1的类别被标记为狗。

    - 接下来，我们需要遍历剩下的三个未标记的锚框：A0、A2和A3。对于A0，与其拥有最大IoU的真实边界框的类别是狗，但IoU低于预定义的阈值（0.5），因此该类别被标记为背景；

    - 对于A2，与其拥有最大IoU的真实边界框的类别是猫，IoU超过阈值，所以类别被标记为猫；

    - 对于A3，与其拥有最大IoU的真实边界框的类别是猫，但值低于阈值，因此该类别被标记为背景
- 结果如下:


```python
labels[2]
```




    tensor([[0, 1, 2, 0, 2]])



- 返回的第二个元素是掩码（mask）变量，形状为（批量大小，锚框数的四倍）。掩码变量中的元素与每个锚框的4个偏移量一一对应。由于我们不关心对背景的检测，负类的偏移量不应影响目标函数。

- 通过元素乘法，掩码变量中的零将在计算目标函数之前过滤掉负类偏移量: 


```python
labels[1]
```




    tensor([[0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 1., 1.,
             1., 1.]])



- 返回的第一个元素包含了为每个锚框标记的四个偏移值。请注意，负类锚框的偏移量被标记为零:


```python
labels[0]
```




    tensor([[-0.00e+00, -0.00e+00, -0.00e+00, -0.00e+00,  1.40e+00,  1.00e+01,
              2.59e+00,  7.18e+00, -1.20e+00,  2.69e-01,  1.68e+00, -1.57e+00,
             -0.00e+00, -0.00e+00, -0.00e+00, -0.00e+00, -5.71e-01, -1.00e+00,
              4.17e-06,  6.26e-01]])



## 4.4 使用非极大值抑制预测边界框

- 在预测时，我们先为图像生成多个锚框，再为这些锚框一一预测类别和偏移量。

- 一个预测好的边界框则根据其中某个带有预测偏移量的锚框而生成。

- offset_inverse函数，该函数将锚框和偏移量预测作为输入，并应用逆偏移变换来返回预测的边界框坐标。


```python
#该函数将锚框和偏移量预测作为输入，并应用逆偏移变换来返回预测的边界框坐标。
#@save
def offset_inverse(anchors, offset_preds):
    """根据带有预测偏移量的锚框来预测边界框"""
    anc = d2l.box_corner_to_center(anchors)
    pred_bbox_xy = (offset_preds[:, :2] * anc[:, 2:] / 10) + anc[:, :2]
    pred_bbox_wh = torch.exp(offset_preds[:, 2:] / 5) * anc[:, 2:]
    pred_bbox = torch.cat((pred_bbox_xy, pred_bbox_wh), axis=1)
    predicted_bbox = d2l.box_center_to_corner(pred_bbox)
    return predicted_bbox
```

- 当有许多锚框时，可能会输出许多相似的具有明显重叠的预测边界框，都围绕着同一目标。

- 为了简化输出，我们可以使用非极大值抑制（non‐maximumsuppression，NMS）合并属于同一目标的类似的预测边界框:

    - 对于一个预测边界框B，目标检测模型会计算每个类别的预测概率。假设最大的预测概率为p，则该概率所对应的类别B即为预测的类别。具体来说，我们将p称为预测边界框B的置信度（confidence）。在同一张图像中，所有预测的非背景边界框都按置信度降序排序，以生成列表L。然后我们通过以下步骤操作排序列表L:

        1. 从L中选取置信度最高的预测边界框B1作为基准，然后将所有与B1的IoU超过预定阈值ϵ的非基准预测边界框从L中移除。这时，L保留了置信度最高的预测边界框，去除了与其太过相似的其他预测边界框。

        2. 从L中选取置信度第二高的预测边界框B2作为又一个基准，然后将所有与B2的IoU大于ϵ的非基准预测边界框从L中移除。

        3. 重复上述过程，直到L中的所有预测边界框都曾被用作基准。此时，L中任意一对预测边界框的IoU都小于阈值ϵ；因此，没有一对边界框过于相似


```python
#@save
def nms(boxes, scores, iou_threshold):
    """对预测边界框的置信度进行排序"""
    B = torch.argsort(scores, dim=-1, descending=True)
    keep = [] # 保留预测边界框的指标
    while B.numel() > 0:
        i = B[0]
        keep.append(i)
        if B.numel() == 1: break
        iou = box_iou(boxes[i, :].reshape(-1, 4), boxes[B[1:], :].reshape(-1, 4)).reshape(-1)
        inds = torch.nonzero(iou <= iou_threshold).reshape(-1)
        B = B[inds + 1]
    return torch.tensor(keep, device=boxes.device)

# 将非极大值抑制应用于预测边界框。
#@save
def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5, pos_threshold=0.009999999):
    """使用非极大值抑制来预测边界框"""
    device, batch_size = cls_probs.device, cls_probs.shape[0]
    anchors = anchors.squeeze(0)
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = []
    for i in range(batch_size):
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
        conf, class_id = torch.max(cls_prob[1:], 0)
        predicted_bb = offset_inverse(anchors, offset_pred)
        keep = nms(predicted_bb, conf, nms_threshold)
        # 找到所有的non_keep索引，并将类设置为背景
        all_idx = torch.arange(num_anchors, dtype=torch.long, device=device)
        combined = torch.cat((keep, all_idx))
        uniques, counts = combined.unique(return_counts=True)
        non_keep = uniques[counts == 1]
        all_id_sorted = torch.cat((keep, non_keep))
        class_id[non_keep] =-1
        class_id = class_id[all_id_sorted]
        conf, predicted_bb = conf[all_id_sorted], predicted_bb[all_id_sorted]
        # pos_threshold是一个用于非背景预测的阈值
        below_min_idx = (conf < pos_threshold)
        class_id[below_min_idx] =-1
        conf[below_min_idx] = 1- conf[below_min_idx]
        pred_info = torch.cat((class_id.unsqueeze(1), conf.unsqueeze(1), predicted_bb), dim=1)
        out.append(pred_info)
    return torch.stack(out)
```

- 将上述算法应用到一个带有四个锚框的具体示例中.为简单起见，我们假设预测的偏移量都是零，这意味着预测的边界框即是锚框。对于背景、狗和猫其中的每个类，我们还定义了它的预测概率。


```python
anchors = torch.tensor([[0.1, 0.08, 0.52, 0.92], [0.08, 0.2, 0.56, 0.95],
                        [0.15, 0.3, 0.62, 0.91], [0.55, 0.2, 0.9, 0.88]])
offset_preds = torch.tensor([0] * anchors.numel())
cls_probs = torch.tensor([[0] * 4, # 背景的预测概率
                [0.9, 0.8, 0.7, 0.1], # 狗的预测概率
                [0.1, 0.2, 0.3, 0.9]]) # 猫的预测概率

# 在图像上绘制这些预测边界框和它们的置信度
fig = d2l.plt.imshow(img)
show_bboxes(fig.axes, anchors * bbox_scale, ['dog=0.9', 'dog=0.8', 'dog=0.7', 'cat=0.9'])
```


    
![svg](img/deeplearning/code/pytorch/12_computer_vision/4_anchor_box_files/4_anchor_box_29_0.svg)
    


- 调用multibox_detection函数来执行非极大值抑制，其中阈值设置为0.5。请注意，我们在示例的张量输入中添加了维度。


```python
output = multibox_detection(cls_probs.unsqueeze(dim=0),
                            offset_preds.unsqueeze(dim=0),
                            anchors.unsqueeze(dim=0),
                            nms_threshold=0.5)
output
```




    tensor([[[ 0.00,  0.90,  0.10,  0.08,  0.52,  0.92],
             [ 1.00,  0.90,  0.55,  0.20,  0.90,  0.88],
             [-1.00,  0.80,  0.08,  0.20,  0.56,  0.95],
             [-1.00,  0.70,  0.15,  0.30,  0.62,  0.91]]])



- 可以看到返回结果的形状是（批量大小，锚框的数量，6）。最内层维度中的六个元素提供了同一预测边界框的输出信息。第一个元素是预测的类索引，从0开始（0代表狗，1代表猫），值‐1表示背景或在非极大值抑制中被移除了。第二个元素是预测的边界框的置信度。其余四个元素是坐标

- 删除‐1类别（背景）的预测边界框后，我们可以输出由非极大值抑制保存的最终预测边界框:


```python
fig = d2l.plt.imshow(img)
for i in output[0].detach().numpy():
    if i[0] == -1:
        continue
    label = ('dog=', 'cat=')[int(i[0])] + str(i[1])
    show_bboxes(fig.axes, [torch.tensor(i[2:]) * bbox_scale], label)
```


    
![svg](img/deeplearning/code/pytorch/12_computer_vision/4_anchor_box_files/4_anchor_box_33_0.svg)
    


- 实践中，在执行非极大值抑制前，我们甚至可以将置信度较低的预测边界框移除，从而减少此算法中的计算量。
