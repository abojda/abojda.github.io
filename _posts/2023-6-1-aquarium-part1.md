---
layout: post
# title: "[#1] Object detection on Aquarium dataset"
# title: "Object detection on Aquarium dataset [#1]"
# title: "Object detection on Aquarium dataset [Part 1]"
title: "Aquarium object detection [Part 1]"
excerpt: # overwrite auto-generated excerpt
---
<!-- TODO - Maybe add here short outline of content of this post (will be good for excerpt in a list of posts) -->
<!-- TODO - Or just add short description in the list of posts? -->
<!-- TODO - Check how others do it! -->
<!-- TODO - Improve title -->
<!-- **Topics covered in part 1:**
 - Project outline
 - Exploratory Data Analysis
 - YOLOv5 baseline -->
**Table of contents**
- [Project motivation and outline](#motivation_outline)
  - [Frameworks and models](#frameworks_models)

<!-- {: include {{site.baseurl}}/_snippets/aquarium.md} -->

## Project motivation and outline {#motivation_outline}
Object detection State-of-The-Art landscape in 2023 seems to be a little unclear. Of course, there are great SoTA leaderboards on [paperswithcode.com](https://paperswithcode.com/), but they only show part of the story.

If we take a look at top 10 models from [Object Detection on COCO test-dev leaderboard](https://paperswithcode.com/sota/object-detection-on-coco) we will notice that most of them are large (>1B parameters) models - often infeasible in practical apllications.

of relatively small dataset, compared to COCO's 330K images.
(arguably) the most popular


**Therefore my main motivation for the project is to explore multiple object detection frameworks in order to gain understanding of their philosophy, possibilities, limitations and available methods/models.**

In the beginning, I'm not planning to focus on running extensive hyperparameters tuning or exploring different training techniques. Doing so for multiple models/frameworks would require access to quite a lot of computing power.

I intend to run more such experiments for some selected models after obtaining baseline results.

### Frameworks and models {#frameworks_models}

### Dataset selection

## Exploratory Data Analysis
<!-- Let's move on and explore the aquarium dataset.  -->
<!-- TODO - put EDA in notebook and link it here (or Quarto?-->
Roboflow ["Health Check"](https://universe.roboflow.com/brad-dwyer/aquarium-combined/health) tab already gives us great insight into the data:

<img src="{{site.baseurl}}/images/posts/aquarium-part1/health-check-1.png" alt="health-check-1" width="100%" height="100%">

- Dataset consists of 638 images with 4821 annotations
- There is only one null image (no objects present and therefore no annotations)
- There are 7 classes representing 7 different aquatic animals 
- **Classes are highly imbalanced**


## Baseline models
### YOLO v5

### YOLO v8
