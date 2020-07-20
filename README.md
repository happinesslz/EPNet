# EPNet
EPNet: Enhancing Point Features with Image Semantics for 3D Object Detection(ECCV 2020)
Paper is now available in [EPNet](https://arxiv.org/pdf/2007.08856.pdf)

Code will be released before the  ECCV 2020 online conference.

## Highlights
0. Without extra image annotations, e.g. 2D bounding box, Semantic labels and so on. 
2. A more accurate multi-scale point-wise fusion for Image and Point Cloud. 
3. The proposed CE loss can improve the performance of 3D Detection greatly.
3. Without GT AUG.

## Contributions
This is Pytorch implementation for EPNet on KITTI dataset, which  is mainly achieved by  [Liu Zhe](https://github.com/happinesslz) and [Huang Tengteng](https://github.com/tengteng95). Some parts also benefit from [Chen Xiwu](https://github.com/XiwuChen).

## Abstract
In this paper, we aim at addressing two critical issues in the 3D detection task, including the exploitation of multiple sensors~(namely LiDAR point cloud and camera image), as well as the inconsistency between the localization and classification confidence. To this end, we propose a novel fusion module to enhance the point features with semantic image features in a point-wise manner without any image annotations. Besides, a consistency enforcing loss is employed to explicitly encourage the consistency of both the localization and classification confidence. We design an end-to-end learnable framework named EPNet to integrate these two components. Extensive experiments on the KITTI and SUN-RGBD datasets demonstrate the superiority of EPNet over the state-of-the-art methods. 

![image](img/1.jpg)

## Network
The architecture of our two-stream RPN is shown in the below.
![image](img/2.jpg)



The architecture of our LI-Fusion module in the two-stream RPN.
![image](img/3.jpg)

## Implementation
To Do







