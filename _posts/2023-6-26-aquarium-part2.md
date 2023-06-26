---
layout: post
title: "Aquarium object detection #2 - YOLOv5 baseline"
excerpt: "In this article we focus on training and improving a baseline model using the YOLOv5 framework. It also introduces basic model evaluation techniques."
---
{% capture imgurl %}{{site.baseurl}}/images/posts/aquarium-part2{% endcapture %}
{% assign part = 2 %}
**Table of contents**
1. [Default YOLOv5 training]({{site.baseurl}}{{page.url}}#default_training)
    1. [Training code]({{site.baseurl}}{{page.url}}#default_training_code)
    2. [Training results]({{site.baseurl}}{{page.url}}#default_training_results)
2. [Training improvement]({{site.baseurl}}{{page.url}}#training_improvement)
    1. [Limiting objectness loss gain]({{site.baseurl}}{{page.url}}#training_improvement_objectness)
    2. [Improved training results]({{site.baseurl}}{{page.url}}#training_improvement_results)
    3. [Longer training]({{site.baseurl}}{{page.url}}#training_improvement_longer)
3. [Model evaluation]({{site.baseurl}}{{page.url}}#model_evaluation)
    1. [Comparison of mean average precisions]({{site.baseurl}}{{page.url}}#model_evaluation_comparison)
    2. [Precision-Recall and F1-score curves]({{site.baseurl}}{{page.url}}#model_evaluation_precision_recall)
    3. [Confusion matrix]({{site.baseurl}}{{page.url}}#model_evaluation_confusion_matrix)
    4. [Predictions vs ground-truth annotations]({{site.baseurl}}{{page.url}}#model_evaluation_predictions_vs_ground_truth)
4. [Summary]({{site.baseurl}}{{page.url}}#summary)

<br/>

{% include_relative snippets/aquarium_series.html part=part %}

<br/>

# 1. Default YOLOv5 training {#default_training}
We will start our object detection journey by developing a baseline model. This model will be our point of reference for further experiments. The aquarium dataset on Roboflow offers access to the [already trained YOLOv5 model](https://universe.roboflow.com/brad-dwyer/aquarium-combined/dataset/5). The model tab reports 74% mAP@0.5 on a validation set and 73% mAP@0.5 on a test set.

<div class="img-container">
  <a href="{{imgurl}}/roboflow-valid-metrics.png" target="_blank">
    <img src="{{imgurl}}/roboflow-valid-metrics.png" alt="Roboflow validation metrics" width="100%" height="100%">
  </a>
  <p> Validation set metrics </p>
</div>

<div class="img-container">
  <a href="{{imgurl}}/roboflow-test-metrics.png" target="_blank">
    <img src="{{imgurl}}/roboflow-test-metrics.png" alt="Roboflow test metrics" width="100%" height="100%">
  </a>
  <p> Test set metrics </p>
</div>


**We'll try to recreate these results as our baseline.**


[YOLOv5](https://github.com/ultralytics/yolov5) is a single-stage object detector released in 2020, which at the time claimed to offer state-of-the-art object detection. Its easy-to-use framework made it a popular choice, especially among practitioners. While YOLOv5 is no longer state-of-the-art, it remains a reasonable out-of-the-box option for establishing a baseline model.

For our training and evaluation notebooks we will use [packaged version of YOLOv5](https://github.com/fcakyon/yolov5-pip), which is a Python wrapper for ultralytics/YOLOv5 scripts. The package is available through pip and offers some additional useful features including integration with HuggingFace Hub.


## 1.1 Training code {#default_training_code}
Training notebook is available [on GitHub](https://github.com/abojda/aquarium-object-detection/blob/main/notebooks/yolov5/aquarium_yolov5_train.ipynb) and in Google Colab: 
<a target="_blank" href="https://colab.research.google.com/github/abojda/aquarium-object-detection/blob/main/notebooks/yolov5/aquarium_yolov5_train.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

### Dataset download
Before the training, we first have to download data. We will again use Roboflow, but this time we will download data already in YOLOv5 format.

```bash
%env ROBOFLOW_API_KEY=#########
!curl -L "https://universe.roboflow.com/ds/aXGylruXWt?key=$ROBOFLOW_API_KEY" > roboflow.zip
!unzip -o -q roboflow.zip -d data && rm roboflow.zip
```

Unfortunately downloaded dataset misses information about the dataset root directory in the `data.yaml` file.
We have to insert this path ourselves in the first line of the file.
```bash
!sed -i "1i path: /content/data" data/data.yaml
```

There is one more thing we have to do before running training. During exploratory data analysis, we found out [that one of the images is mislabeled]({{site.baseurl}}/aquarium-part1#mislabeled_image). Let's correct the annotations for this image. We will simply replace all category ids with `1` (jellyfish) as the image contains only jellyfish objects.

```bash
%env mislabeled_file=data/train/labels/IMG_8590_MOV-3_jpg.rf.e215fd21f9f69e42089d252c40cc2608.txt
!awk '{print "1", $2, $3, $4, $5}' $mislabeled_file > tmp.txt && mv tmp.txt $mislabeled_file
```

### Training
Running training requires just a few lines of code.

```python
from yolov5 import train

train.run(imgsz=640,
          epochs=300,
          data='data/data.yaml',
          weights='yolov5s.pt',
          logger='TensorBoard',
          cache='ram');
```

With this code we will train a "small" model starting from COCO-pretrained weights (`yolov5s.pt`), using default hyperparameters. The image size is set to 640, which is also a default. Training will run for 300 epochs. TensorBoard was selected as a logger, so we can run it in a notebook to monitor the progress of the training.

```bash
%load_ext tensorboard
%tensorboard --logdir runs/train
```

**We also opted to cache images in memory with the `cache='ram'` option as it drastically shortens training time. Training of a small model for 100 epochs takes over 3 hours without caching and around 18 minutes with memory caching -- that is 10x faster!**


## 1.2 Training results {#default_training_results}
Results of our first training are available in `runs/train/exp` directory. Let's compare our "small" model training charts with training charts for the Roboflow model.

### Our model
<div class="img-container">
  <a href="{{imgurl}}/training-graphs/default-s.png" target="_blank">
    <img src="{{imgurl}}/training-graphs/default-s.png" alt="Small model training charts" width="100%" height="100%">
  </a>
</div>

### Roboflow model
<div class="img-container">
  <a href="{{imgurl}}/training-graphs/default-roboflow.png" target="_blank">
    <img src="{{imgurl}}/training-graphs/default-roboflow.png" alt="Roboflow training charts" width="100%" height="100%">
  </a>
</div>

We can see that both plots are very similar. Let's also compare the validation set mAP@0.5 metrics reported at the end of the training to get a better understanding of the results.

{% include_relative snippets/aquarium-part2/first-val-map-table.html %}

We managed to recreate baseline results from the Roboflow model and obtained even better results out-of-the-box, with default training settings:
- The overall mAP improved by over five percentage points
- We also see an improvement in the metrics for each of the classes except for the puffin
- The model seems to be undertrained and it looks like there is room for improvement with longer training

I would attribute such significant improvement to one of the two things. Roboflow model could be an even smaller, less capable, YOLOv5n ("nano") model -- unfortunately, I haven't found information about model size on the Roboflow page. The second option is that both models are "small" models and improvements come from enhanced training routine, better data augmentation, modified default hyperparameters, or other changes introduced with new YOLOv5 releases.

We used validation set metrics as a first point of reference for model performance. This makes sense since we didn't use a validation set to tune the hyperparameters. Later on, we will conduct more detailed evaluations with the test set.

# 2. Training improvement {#training_improvement}
## 2.1 Limiting objectness loss gain {#training_improvement_objectness}
There is one more clear conclusion coming from the above training results -- validation objectness loss is overfitting from early epochs, while other loss components are still decreasing. YOLOv5 loss function consists of three weighted components:
- Location (box) loss
- Objectness loss
- Classes loss

**To eliminate observed overfitting we can try to decrease the weight (gain) associated with the objectness component**. In YOLOv5 objectness loss gain is defined as `obj` hyperparameter. Let's first update our training script to allow the modification of hyperparameters.

YOLOv5 uses `.yaml` files to store hyperparameter config. Default hyperparameters are defined in [`hyp.scratch-low.yaml` file](https://github.com/ultralytics/yolov5/blob/master/data/hyps/hyp.scratch-low.yaml). We could just manually download and modify this file, however, let's develop a solution that doesn't require any manual file manipulation. By following this approach, we can eliminate the need for manual file uploads in Google Colab and use Python code to modify the default config.

```python
import yaml
import torch

# Default hyperparameters config
hyp_file = 'hyp.scratch-low.yaml'
hyp_url = f'https://raw.githubusercontent.com/ultralytics/yolov5/master/data/hyps/{hyp_file}'

# Get default hyperparameters config
torch.hub.download_url_to_file(hyp_url, hyp_file)

# Load YAML into dict
with open(hyp_file, errors='ignore') as f:
    hyps = yaml.safe_load(f)

# MODIFY HYPERPARAMETERS
hyps['obj'] = 0.3

# Dump dict into YAML file
with open(hyp_file, 'w') as f:
  yaml.dump(hyps, f, sort_keys=False)
```

Default hyperparameters are loaded into the `hyps` dictionary and can be modified there. Dictionary is later dumped back to the `.yaml` file. If no changes are applied, then we just train with the default config from `hyp.scratch-low.yaml`.

To run the training with modified hyperparameters we just need to pass the `hyp` argument to `train.run()`.
```python
train.run(imgsz=640,
          epochs=300,
          data='data/data.yaml',
          weights='yolov5s.pt',
          logger='TensorBoard',
          cache='ram',
          hyp=hyp_file)
```


## 2.2 Improved training results {#training_improvement_results}
Let's now train with an arbitrarily selected objectness gain value of 0.3 (default value is 1.0). The result of this run is presented below, with the default run (`results_s_default`) as a reference.

<div class="img-container">
  <a href="{{imgurl}}/training-graphs/obj-gain-0.3-loss.png" target="_blank">
    <img src="{{imgurl}}/training-graphs/obj-gain-0.3-loss.png" width="100%" height="100%">
  </a>
</div>

<div class="img-container">
  <a href="{{imgurl}}/training-graphs/obj-gain-0.3-map.png" target="_blank">
    <img src="{{imgurl}}/training-graphs/obj-gain-0.3-map.png" width="100%" height="100%">
  </a>
</div>

It looks like objectness loss is still overfitting. Let's run another training with an even smaller gain of 0.1

<div class="img-container">
  <a href="{{imgurl}}/training-graphs/obj-gain-0.1-loss.png" target="_blank">
    <img src="{{imgurl}}/training-graphs/obj-gain-0.1-loss.png" width="100%" height="100%">
  </a>
</div>

<div class="img-container">
  <a href="{{imgurl}}/training-graphs/obj-gain-0.1-map.png" target="_blank">
    <img src="{{imgurl}}/training-graphs/obj-gain-0.1-map.png" width="100%" height="100%">
  </a>
</div>

This experiment looks more promising -- we've mitigated the overfitting issue, and validation mAP@0.5 after 300 epochs has improved.

However, it also looks like the models are still undertrained and could benefit from longer training:
- Validation loss (box/cls) is still decreasing
- Validation mAP keeps increasing



## 2.3 Longer training {#training_improvement_longer}
Let's train the two models (default and one with `obj=0.1`) for another 100 epochs and see if they can improve further. To continue training we can just supply `last.pt` or `best.pt` weights from previous trainings to a new `train.run()` call -- new training will start from our already trained weights instead of COCO-pretrained weights.
```python
train.run(imgsz=640,
          epochs=100,
          data='data/data.yaml',
          weights='best.pt',
          logger='TensorBoard',
          cache='ram',
          hyp=hyp_file)
```

There is however one more issue. By default YOLOv5 applies warmup epochs and warmup bias in its learning rate scheduler.
```yaml
# hyp.scratch-low.yaml
...
warmup_epochs: 3.0  # warmup epochs (fractions ok)
warmup_momentum: 0.8  # warmup initial momentum
warmup_bias_lr: 0.1  # warmup initial bias lr
...
```

If we continue training with these hyperparameters we will get suboptimal results. The initial learning rate is very high, and therefore training is very unstable -- in a sense, the model forgot part of the previous training.

<a href="{{imgurl}}/training-graphs/warmup-issue.png" target="_blank">
  <img src="{{imgurl}}/training-graphs/warmup-issue.png" width="100%" height="100%">
</a>

Our problem comes from the `warmup_bias_lr: 0.1` hyperparameter, which sets initial learning rate to 0.1. We can just change it to 0.0 in our training script in the same way as with `obj` hyperparameter before.
```python
hyps['warmup_bias_lr'] = 0.0
```

### Longer training without warmup bias
Let's now run longer trainings without warmup bias. We will also lower the learning rate to `lr=0.001` (from default `lr=0.01`) to avoid problems with stability.

```python
hyps['warmup_bias_lr'] = 0.0
hyps['lr0'] = 0.001
```
Results of these trainings are presented below.

<div class="img-container">
  <a href="{{imgurl}}/training-graphs/comparison-loss.png" target="_blank">
    <img src="{{imgurl}}/training-graphs/comparison-loss.png" width="100%" height="100%">
  </a>
</div>

<div class="img-container">
  <a href="{{imgurl}}/training-graphs/comparison-map.png" target="_blank">
    <img src="{{imgurl}}/training-graphs/comparison-map.png" width="100%" height="100%">
  </a>
</div>

We can see that mAP@0.5 for both models hasn't really improved, and neither did mAP@0.5:0.95 for `obj=0.1` model. In contrast, the default model saw a slight improvement in mAP@0.5:0.95, with a further increasing trend. Let's also compare per-class mAP values on a validation set for `best.pt` models after 300 and 400 epochs.

{% include_relative snippets/aquarium-part2/val-map-table.html %}


It seems that `obj=0.1` models generally perform better and that longer training brought only little if any improvement. We could of course try to train these models for even longer or experiment with different hyperparameters. But let's finish the training here and move to the model evaluation, where we will compare both models on an unbiased test set to get a better understanding of their performance and the ability to generalize to unseen data.


# 3. Model evaluation {#model_evaluation}
Before we start evaluation, it is important to note that our test set is relatively small. Comparison of test-set metrics might not reflect the true performance of the models on the unseen data. This is particularly noticeable for less numerous classes, where changes in metrics might be abrupt -- for example, there are only 11 starfish objects in our test set.

YOLOv5 has an evaluation script that calculates the most important metrics out-of-the-box. A simple evaluation notebook can be found [on GitHub](https://github.com/abojda/aquarium-object-detection/blob/main/notebooks/yolov5/aquarium_yolov5_eval.ipynb) and in Google Colab: 
<a target="_blank" href="https://colab.research.google.com/github/abojda/aquarium-object-detection/blob/main/notebooks/yolov5/aquarium_yolov5_eval.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
.
We download the Roboflow dataset with the same code as in the training notebook, and then we can just run the evaluation, selecting data split with the `task` keyword.

```python
from yolov5 import val

weights = 'best.pt'
# weights = 'akbojda/yolov5s-aquarium'

val.run(imgsz=640,
        data='data/data.yaml',
        weights=weights,
        task='test')
```
Of course, we also have to provide the weights of the model that we want to evaluate. One way is to just pass the path to the local `.pt` file as we did before. The other interesting option is to upload the model to the [HuggingFace Model Hub](https://huggingface.co/models). Packaged YOLOv5 has full HuggingFace Hub integration -- we can just use the model name and it will be downloaded under the hood. For example, `weights = 'akbojda/yolov5s-aquarium'` will use [this model](https://huggingface.co/akbojda/yolov5s-aquarium) that I uploaded to the 🤗 Hub.


## 3.1 Comparison of mean average precisions {#model_evaluation_comparison}
Let's first take a look at the values of mAP@0.5 and mAP@0.5:0.95 calculated on the test dataset, for different models.

{% include_relative snippets/aquarium-part2/test-map0.5-table.html %}

<br/>

{% include_relative snippets/aquarium-part2/test-map-table.html %}

<br/>

Clearly, models trained with limited objectness gain (`obj=0.1`) perform better compared to the default training configuration. It also seems that extended training hasn't improved the overall performance. In conclusion: the `obj=0.1` model trained for 300 epochs looks to be the best choice.

## 3.2 Precision-Recall and F1-score curves {#model_evaluation_precision_recall}
Let's take a closer look at other metrics of this model, starting with the evaluation summary generated with the `val.run` call. mAP values are identical as in the comparison tables above. Precision and recall values are reported at the maximum F1-score confidence threshold.
{% include_relative snippets/aquarium-part2/test_s_obj0.1_300_results.html %}

<br/>

The validation script also plots Precision-Recall, Precision-Confidence, Recall-Confidence, and F1-Confidence curves.

<a href="{{imgurl}}/test_s_obj0.1_300/PR_curve.png" target="_blank">
  <img src="{{imgurl}}/test_s_obj0.1_300/PR_curve.png" width="100%" height="100%">
</a>

<a href="{{imgurl}}/test_s_obj0.1_300/P_curve.png" target="_blank">
  <img src="{{imgurl}}/test_s_obj0.1_300/P_curve.png" width="100%" height="100%">
</a>

<a href="{{imgurl}}/test_s_obj0.1_300/R_curve.png" target="_blank">
  <img src="{{imgurl}}/test_s_obj0.1_300/R_curve.png" width="100%" height="100%">
</a>

<a href="{{imgurl}}/test_s_obj0.1_300/F1_curve.png" target="_blank">
  <img src="{{imgurl}}/test_s_obj0.1_300/F1_curve.png" width="100%" height="100%">
</a>

In object detection we always face a [trade-off between precision and recall](https://medium.com/analytics-vidhya/precision-recall-tradeoff-for-real-world-use-cases-c6de4fabbcd0):
- When we cannot afford to miss any detection, we look for high recall
- When we cannot afford to have any incorrect detection we look for high precision

We can use the above charts to select a confidence threshold that gives us the desired trade-off between precision and recall. If we don't have a strong preference for one of the metrics, we can use the F1-score, which is the harmonic mean of precision and recall. On the F1-Confidence curve, we can see that the highest F1 score (combined for all classes) has a value of 0.82, and is obtained at a 0.444 confidence threshold. We can also find this threshold in the Precision-Confidence and Recall-Confidence charts and confirm values reported in the evaluation summary table -- precision=0.897 and recall=0.756.

It is also worth noting that we obtain very similar F1 scores for confidence thresholds in a range between 0.15 and 0.65 -- these values seem to be good threshold candidates if we want to optimize recall or precision respectively.

Let's compare precision and recall values at these three different confidence thresholds.
<table class="center">
  <tr>
    <th style="text-align: center">Confidence threshold</th>
    <th style="text-align: center">Precision</th>
    <th style="text-align: center">Recall</th>
  </tr>
  <tr>
    <td style="text-align: center">0.15</td>
    <td style="text-align: center">0.78</td>
    <td style="text-align: center">0.805</td>
  </tr>
  <tr>
    <td style="text-align: center">0.444</td>
    <td style="text-align: center">0.897</td>
    <td style="text-align: center">0.756</td>
  </tr>
  <tr>
    <td style="text-align: center">0.65</td>
    <td style="text-align: center">0.945</td>
    <td style="text-align: center">0.71</td>
  </tr>
</table>

We can observe the precision-recall trade-off that comes with these different thresholds. For our detection task threshold of 0.444 seems to be a good choice as we don't have a strong preference for either precision or recall, and want balanced performance.

## 3.3 Confusion matrix {#model_evaluation_confusion_matrix}
Let's also take a look at the confusion matrix.

<a href="{{imgurl}}/test_s_obj0.1_300/confusion_matrix_0.444.png" target="_blank">
  <img src="{{imgurl}}/test_s_obj0.1_300/confusion_matrix_0.444.png" width="100%" height="100%">
</a>

The confusion matrix won't be very useful for analyzing false negatives and false positives -- we already did it by looking at precision and recall curves (overall and per-class). But we can use this matrix to verify some conclusions we came up with during exploratory data analysis (part 1 of the series) and make some new observations:
<table class="center">
  <tr>
    <th style="text-align: center">EDA observation</th>
    <th style="text-align: center">Confusion matrix observation</th>
  </tr>
  <tr>
    <td style="text-align: center">"Fish, sharks, and stingrays are relatively similar animals, which might be misidentified, especially in adverse conditions"</td>
    <td style="text-align: center; color:green"> Partially true &ndash; we can see that sharks are sometimes detected as fish</td>
  </tr>
  <tr>
    <td style="text-align: center">"Penguins and puffins are relatively similar animals, which might be misidentified"</td>
    <td style="text-align: center; color:red"> False &ndash; we can see that these classes aren't misidentified </td>
  </tr>
  <tr>
    <td style="text-align: center">"There are two classes, which are quite distinctive and not similar to the others: jellyfish and starfish"</td>
    <td style="text-align: center; color:orange"> 50/50 &ndash; jellyfish and starfish aren't actually misidentified, but stingrays are identified as jellyfish in some situations</td>
  </tr>
</table>


## 3.4 Predictions vs ground-truth annotations {#model_evaluation_predictions_vs_ground_truth}
The last thing we will do as part of the model validation will be to look at the predictions for specific images and compare them to ground-truth annotations. The validation script already outputs some detection results in the form of mosaics -- **ground truth annotations are on the left and our model predictions are on the right**.

<div class="img-container">
  <a href="{{imgurl}}/test_s_obj0.1_300/val_batch0_labels.jpg" target="_blank">
    <img src="{{imgurl}}/test_s_obj0.1_300/val_batch0_labels.jpg" width="48%" height="48%">
  </a>

  <a href="{{imgurl}}/test_s_obj0.1_300/val_batch0_pred.jpg" target="_blank">
    <img src="{{imgurl}}/test_s_obj0.1_300/val_batch0_pred.jpg" width="48%" height="48%">
  </a>
</div>

<div class="img-container">
  <a href="{{imgurl}}/test_s_obj0.1_300/val_batch1_labels.jpg" target="_blank">
    <img src="{{imgurl}}/test_s_obj0.1_300/val_batch1_labels.jpg" width="48%" height="48%">
  </a>

  <a href="{{imgurl}}/test_s_obj0.1_300/val_batch1_pred.jpg" target="_blank">
    <img src="{{imgurl}}/test_s_obj0.1_300/val_batch1_pred.jpg" width="48%" height="48%">
  </a>
</div>

However, these mosaics contain only part of the test set. Also, images containing multiple objects are unreadable. Let's instead download model predictions in COCO-json format -- we can do it with the `save_json=True` argument passed to validation run.

```python
val.run(imgsz=640,
        data='data/data.yaml',
        weights=weights,
        task='test',
        save_json=True)
```

Unfortunately, [exported file](https://github.com/abojda/aquarium-object-detection/blob/main/results/yolov5/evaluation/test_s_obj0.1_300/predictions_yolov5.json) contains only a list of annotations, and misses information about images or metadata. Moreover, `image_id` field in annotations contains the filename instead of the image id as defined in [COCO data format](https://cocodataset.org/#format-data). 
To fix this, I first copied missing information from the ground-truth annotation file (and removed the "creatures" supercategory from `categories` list). Then, I used the following script to rewrite the `image_id` field in annotations.


```python
import json

def get_image_id(name, images):
    for image in images:
        if image['file_name'] == f'{name}.jpg':
            return image['id']

    raise ValueError(name)

if __name__ == '__main__':
    with open('predictions_fixed.json', 'r') as in_f:
        data = json.load(in_f)

    for ann in data['annotations']:
        ann['image_id'] = get_image_id(ann['image_id'], data['images'])

    with open('predictions_fixed.json', 'w') as out_f:
        json.dump(data, out_f)
```

The resulting file can be seen [here](https://github.com/abojda/aquarium-object-detection/blob/main/results/yolov5/evaluation/test_s_obj0.1_300/predictions_fixed.json).

### Predictions analysis with FiftyOne
To compare ground-truth annotations with model predictions we will use FiftyOne -- the library that we already used during exploratory analysis. Notebook with this part of the evaluation can be found [on GitHub](https://github.com/abojda/aquarium-object-detection/blob/main/notebooks/aquarium_fiftyone_eval.ipynb) and in Google Colab:
<a target="_blank" href="https://colab.research.google.com/github/abojda/aquarium-object-detection/blob/main/notebooks/aquarium_fiftyone_eval.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
.
Loading ground-truth annotations and predictions is straightforward.

```python
import fiftyone as fo

dataset = fo.Dataset.from_dir(
    name='Aquarium Combined',
    dataset_type=fo.types.COCODetectionDataset,
    data_path='test',
    labels_path='test/_annotations.coco.json',
    label_field='ground_truth',
)

pred_dataset = fo.Dataset.from_dir(
    dataset_type=fo.types.COCODetectionDataset,
    data_path='test',
    labels_path='predictions_fixed.json',
    label_field='model',
)

dataset.merge_samples(pred_dataset)

session = fo.launch_app(dataset)
```

Now, we can easily use FiftyOne to i.e. compare ground-truth annotations with model predictions, at different confidence levels.

**TODO - add video**

# 4. Summary {#summary}
In this article, we managed to establish a baseline model, tested some straightforward improvement ideas, and evaluated trained models. There are of course other aspects we could explore, including running experiments with larger (m/l/x) models or doing hyperparameter optimization. But we will end here as we’ve achieved our two main goals of understanding the YOLOv5 framework and establishing a baseline model.

<br/>

## Further reading
{% include_relative snippets/aquarium_series.html part=part %}
