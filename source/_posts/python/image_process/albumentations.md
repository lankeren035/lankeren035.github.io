---
title: python中albumentations库
date: 2025-02-12 7:00:00
toc: true
tags: [python]
categories: [python]

---

#  

<!-- more -->

# 1. 图像处理

- 对输入图片进行resize, 中心裁剪：

```python
import albumentations

self.rescaler = albumentations.SmallestMaxSize(max_size=sample_size) #_保持最大边长不超过sample_size
self.cropper =albumentations.CenterCrop(height=sample_size, width=sample_size)
self.transforms = [self.rescaler, self.cropper]
self.preprocessor = albumentations.Compose(transforms=self.transforms)#将上面的变换组合起来

image = Image.open(image_path_list[index])
if not image.mode == "RGB":
    image = image.convert("RGB")
    try:
        image = np.array(image).astype(np.uint8)
    except: 
        print(image_path_list[index])
image = self.preprocessor(image=image)["image"]
```

