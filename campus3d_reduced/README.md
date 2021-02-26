# Campus3D
## Introduction
## Installation
## Training
### Train from scratch

  Check the configuration files in config/ and run experiments, eg:
### Train from pretrained model
## Evaluation
## MODEL ZOO
### MODELS
|Model|Method|MC Level|Training Process|Scheduler|Dropout<br>Rate|Download<br>Link|
:-:|:-:|:-:|:-:|:-:|:-:|:-:
|Pointnet2|MC|0|50 epochs(lr=0.01)|cos|0.5|[MC0]()|
|Pointnet2|MC|1|50 epochs(lr=0.01)|cos|0.5|[MC1]()|
|Pointnet2|MC|2|50 epochs(lr=0.01)|cos|0.5|[MC2]()|
|Pointnet2|MC|3|50 epochs(lr=0.01)|cos|0.5|[MC3]()|
|Pointnet2|MC|4|50 epochs(lr=0.01)|cos|0.5|[MC4]()|
|Pointnet2|MTnc|-1|50 epochs(lr=0.01)|cos|0.5|[pointnet2_MTnc]()|
|Pointnet2|MT|-1|50 epochs(lr=0.01) +<br>20 epochs(lr=0.01)|cos|0.5|[pointnet2_MT]()|
|DGCNN|MT|-1|50 epochs(lr=0.01) +<br>20 epochs(lr=0.01)|cos|0.5|[dgcnn_MT]()|
|PointCNN|MT|-1|50 epochs(lr=0.01) +<br>30 epochs(lr=0.01)|cos|0.5|[pointcnn_MT]()|
### SEMANTIC SEGMENTATION
|Granularity Level|Class|MC|MC+HE|MTnc|MT|MT+HE|
-:|:-:|:-:|:-:|:-:|:-:|:-
