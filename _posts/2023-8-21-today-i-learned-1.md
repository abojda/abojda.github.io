---
layout: post
title: "PyTorch collate_fn with different data types [TIL #1]"
excerpt: "In this article, we investigate how PyTorch DataLoader collates different types of data and collections by default. We also learn how to implement a custom collate function for our specific data format."
---

**Table of contents**
1. [General examples]({{site.baseurl}}{{page.url}}#general_examples)
    1. [Returning image Tensor and int target]({{site.baseurl}}{{page.url}}#general_int)
    2. [Returning image Tensor and dictionary target]({{site.baseurl}}{{page.url}}#general_dict)
    3. [Custom collate_fn]({{site.baseurl}}{{page.url}}#custom_collate_fn)
2. [Real-life example - torchvision object detection]({{site.baseurl}}{{page.url}}#real_life_example)
    1. [Motivation]({{site.baseurl}}{{page.url}}#motivation)
    2. [Initial implementation]({{site.baseurl}}{{page.url}}#initial_implementation)
    3. [Solution]({{site.baseurl}}{{page.url}}#solution)

<br/>

{% include_relative snippets/til_series.html part=part %}

<br/>

# Introduction

In this article, we investigate how PyTorch DataLoader collates different types of data and collections by default
- We focus on dictionaries, but the same steps allow us to understand the behavior of any type of data
- We also learn how to implement a custom collate function for our specific data format



# TL;DR
- Default collate behavior in PyTorch DataLoader depends on the type of the object/collection returned from the PyTorch Dataset
- By default, DataLoader uses the [default_collate](https://github.com/pytorch/pytorch/blob/v2.0.1/torch/utils/data/_utils/collate.py#L204) function to collate lists of samples into batches
- To check how different data types are handled by `default_collate` we can investigate [examples in the docstring of this function](https://github.com/pytorch/pytorch/blob/v2.0.1/torch/utils/data/_utils/collate.py#L230)
- It is also possible to write custom `collate_fn` - examples in sections 1.3 and 2.3 below

<br/>

# 1. General examples {#general_examples}
Code is also available as a Colab Notebook:
<a target="_blank" href="https://colab.research.google.com/github/abojda/today-i-learned/blob/main/pytorch/1.dataloader_collate_fn.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

The examples below all use 2x2 RGB images for simplicity - e.g. `torch.rand(3, 2, 2)`.

## 1.1 Returning image Tensor and int target {#general_int}
```python
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class TensorIntDataset(Dataset):
    def __init__(self, n_samples):
        self.imgs = [torch.rand(3, 2, 2) for i in range(n_samples)]
        self.targets = np.random.randint(0, 9, size=n_samples)

    def __getitem__(self, idx):
        return self.imgs[idx], self.targets[idx]

    def __len__(self):
        return len(self.targets)


ti_dataset = TensorIntDataset(10)
ti_dataloader = DataLoader(ti_dataset, batch_size=2)
```

A single sample from this dataset is an image tensor in CHW format and `int` target.

```python
img, target = ti_dataset[0]

print(img.shape)
# torch.Size([3, 2, 2])

print(target)
# 3
```

So, as we can expect the Dataloader will return the batch of images in NCHW format and the tensor with targets.

```python
imgs, targets = next(iter(ti_dataloader))

print(imgs.shape)
# torch.Size([2, 3, 2, 2])

print(targets)
# tensor([3, 5])
```

## 1.2 Returning image Tensor and dictionary target {#general_dict}

```python
class TensorDictDataset(Dataset):
    def __init__(self, n_samples):
        self.imgs = [torch.rand(3, 2, 2) for i in range(n_samples)]
        self.targets = [
            {
              "label": np.random.randint(0, 9),
              "other_value": np.random.randint(0, 9),
            }
            for i in range(n_samples)
        ]

    def __getitem__(self, idx):
        return self.imgs[idx], self.targets[idx]

    def __len__(self):
        return len(self.targets)
```

A single sample from this dataset is an image tensor in CHW format and the target dictionary.

```python
td_dataset = TensorDictDataset(10)
img, target = td_dataset[0]

print(img.shape)
# torch.Size([3, 2, 2])

print(target)
# {'label': 1, 'other_value': 2}
```

So, based on example 1.1, we might expect that the Dataloader will return the target as a list of dictionaries - for example:

```python
targets = [
    {
        "label": 4,
        "other_value": 0,
    },
    {
        "label": 2,
        "other_value": 6,
    },
]
```

However, this is not the case!

In fact, the Dataloader will return the batch of images in NCHW format and the **single target dictionary containing targets for all the samples**.

```python
td_dataloader = DataLoader(td_dataset, batch_size=2)
imgs, targets = next(iter(td_dataloader))

print(imgs.shape)
# torch.Size([2, 3, 2, 2])

print(targets)
# {'label': tensor([1, 5]), 'other_value': tensor([2, 3])}
```

**By default, DataLoader uses the [default_collate](https://github.com/pytorch/pytorch/blob/v2.0.1/torch/utils/data/_utils/collate.py#L204) function to collate lists of samples into batches.**

To check how different data types are handled by `default_collate` we can investigate the docstring of this function - for example, behavior for `Mapping` is described [here](https://github.com/pytorch/pytorch/blob/v2.0.1/torch/utils/data/_utils/collate.py#L238) and we can see that it matches the output format we obtained above.


## 1.3 Custom collate_fn {#custom_collate_fn}
To modify collate behavior for our specific needs we can write custom collate function based on the [hint from the docstring](https://github.com/pytorch/pytorch/blob/v2.0.1/torch/utils/data/_utils/collate.py#L251).

```python
def custom_collate(batch):
    if (
        isinstance(batch, list)
        and len(batch[0]) == 2
        and isinstance(batch[0][1], dict)
    ):
        imgs = torch.stack([img for img, target in batch])
        targets = [target for img, target in batch]
        return imgs, targets
    else:  # Fall back to `default_collate`
        return torch.utils.data.default_collate(batch)
```

```python
td_dataloader = DataLoader(td_dataset, batch_size=2, collate_fn=custom_collate)
imgs, targets = next(iter(td_dataloader))

print(imgs.shape)
# torch.Size([2, 3, 2, 2])

print(targets)
# [{'label': 1, 'other_value': 2}, {'label': 5, 'other_value': 3}]
```


# 2. Real-life example - torchvision object detection {#real_life_example}
# 2.1 Motivation {#motivation}
Let's imagine the following situation. We work with a `fasterrcnn_resnet50_fpn` object detection model from the torchvision library.

During training, the [model](https://pytorch.org/vision/0.15/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html) expects both the input tensors and a list of target dictionaries containing ground-truth boxes and labels with the following format.

```python
def rand_boxes(n):
    """
    Generate "random" bounding boxes ensuring x2>x1 and y2>y1
    Only for presentation purposes
    """
    xy1 = 0.9 * torch.rand(n, 2)
    xy2 = xy1 + 0.1

    return torch.cat([xy1, xy2], dim=1)

imgs = [torch.rand(3, 2, 2), torch.rand(3, 2, 2)]  # 2 RGB images (2x2 size)
targets = [
    {
        # Ground-truth for the first image
        # 5 boxes with [x1, y1, x2, y2] coordinates and 5 COCO class labels
        "boxes": rand_boxes(5),  # torch.Size([5, 4])
        "labels": torch.randint(low=0, high=91, size=(5,)),
    },
    {
        # Ground-truth for the second image
        # 7 boxes with [x1, y1, x2, y2] coordinates and 7 COCO class labels
        "boxes": rand_boxes(7),  # torch.Size([7, 4])
        "labels": torch.randint(low=0, high=91, size=(7,)),
    },
]
```

```python
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)

model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

# Test if model accepts the data format
model.train()
model(imgs, targets) # OK - valid format
```

## 2.2 Initial implementation {#initial_implementation}

Let's prepare the PyTorch dataset that will return the data in this format

```python
def rand_target():
    n_objects = np.random.randint(1, 10)
    target = {
        "boxes": rand_boxes(n_objects),
        "labels": torch.randint(low=0, high=91, size=(n_objects,)),
    }
    return target


class DetectionDataset(torch.utils.data.Dataset):
    def __init__(self, n_samples):
        self.imgs = [torch.rand(3, 2, 2) for i in range(n_samples)]
        self.targets = [rand_target() for i in range(n_samples)]

    def __getitem__(self, idx):
        return self.imgs[idx], self.targets[idx]

    def __len__(self):
        return len(self.targets)
```

```python
detection_dataset = DetectionDataset(10)

img, target = detection_dataset[0]

print(img.shape)
# torch.Size([3, 2, 2])

print(target)
# {'boxes': tensor([[0.5160, 0.4331, 0.6160, 0.5331],
#         [0.0248, 0.1283, 0.1248, 0.2283],
#         [0.1320, 0.6417, 0.2320, 0.7417],
#         [0.5979, 0.1503, 0.6979, 0.2503],
#         [0.4593, 0.6163, 0.5593, 0.7163],
#         [0.2641, 0.8964, 0.3641, 0.9964],
#         [0.3178, 0.7418, 0.4178, 0.8418]]),
# 'labels': tensor([12, 56, 19, 37, 83, 23, 79])}
```

A single sample returned from the dataset matches the format required by the `torchvision` model

However, if we use the DataLoader with the default collate function, the format of the batched data will be incorrect or we might even encounter `RuntimeError` if the number of targets is different for each sample


```python
detection_dataloader = DataLoader(detection_dataset, batch_size=2)

# This code will throw RuntimeError or TypeError due to the data format problems

# imgs, targets = next(iter(detection_dataloader))
# model.train()
# model(imgs, targets)
```

## 2.3 Solution {#solution}

The solution is to use the custom collate function similar to `custom_collate` we introduced in section 1.3.

```python
def custom_detection_collate(batch):
    if (
        isinstance(batch, list)
        and len(batch[0]) == 2
        and isinstance(batch[0][1], dict)
    ):
        imgs = [img for img, target in batch]
        targets = [target for img, target in batch]
        return imgs, targets
    else:  # Fall back to `default_collate`
        return torch.utils.data.default_collate(batch)


detection_dataloader = DataLoader(
    detection_dataset, batch_size=2, collate_fn=custom_detection_collate
)
```

```python
imgs, targets = next(iter(detection_dataloader))

print(f"{len(imgs)} images of size: {imgs[0].shape}")
# 2 images of size: torch.Size([3, 2, 2])

print(targets)
# [{'boxes': tensor([[0.5160, 0.4331, 0.6160, 0.5331],
#         [0.0248, 0.1283, 0.1248, 0.2283],
#         [0.1320, 0.6417, 0.2320, 0.7417],
#         [0.5979, 0.1503, 0.6979, 0.2503],
#         [0.4593, 0.6163, 0.5593, 0.7163],
#         [0.2641, 0.8964, 0.3641, 0.9964],
#         [0.3178, 0.7418, 0.4178, 0.8418]]),
# 'labels': tensor([12, 56, 19, 37, 83, 23, 79])},
# {'boxes': tensor([[0.2821, 0.0579, 0.3821, 0.1579],
#         [0.7718, 0.6386, 0.8718, 0.7386]]),
# 'labels': tensor([81,  7])}]
```

```python
model.train()
model(imgs, targets) # OK - valid format
```
