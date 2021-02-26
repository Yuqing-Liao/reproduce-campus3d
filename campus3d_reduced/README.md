# Campus3D
## Introduction
## Installation
## Training
### Train from scratch

  Check the configuration files in config/ and run experiments, eg:
### Train from pretrained model
## Evaluation
## MODEL ZOO
### Models
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
### Test results for semantic segmentation 
|Granularity Level|Class|MC|MC+HE|MTnc|MT|MT+HE|
:-:|:-:|:-:|:-:|:-:|:-:|:-:
|C1|ground|85.4|||||
|C1|construction|79.9|||||
|C2|natural|81.1|||||
|C2|man_made|58.5|||||
|C2|construction|78.8|||||
|C3|natural|79.2|||||
|C3|play_field|62.9|||||
|C3|path&stair|8.7|||||
|C3|driving_road|58.4|||||
|C3|construction|76.6|||||
|C4|natural|81.0|||||
|C4|play_field|57.3|||||
|C4|path&stair|9.3|||||
|C4|vehicle|16.6|||||
|C4|not vehicle|57.9|||||
|C4|building|76.5|||||
|C4|link|0.0|||||
|C4|facility|0.0|||||
|C5|natural|80.1|||||
|C5|play_field|52.7|||||
|C5|sheltered|10.6|||||
|C5|unsheltered|7.6|||||
|C5|bus_stop|0.0|||||
|C5|car|18.1|||||
|C5|bus|0.0|||||
|C5|not vehicle|58.1|||||
|C5|wall|46.5|||||
|C5|roof|43.1|||||
|C5|link|0.2|||||
|C5|artificial_landscape|0.0|||||
|C5|lamp|0.0|||||
|C5|others|0.0|||||
