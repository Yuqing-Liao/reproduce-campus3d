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
### Test results of semantic segmentation 
|Granularity Level|Class|MC|MC+HE|MTnc|MT|MT+HE|
:-:|:-:|:-:|:-:|:-:|:-:|:-:
|C1|ground||||||
|C1|construction||||||
|C2|natural||||||
|C2|man_made||||||
|C2|construction||||||
|C3|natural||||||
|C3|play_field||||||
|C3|path&stair||||||
|C3|driving_road||||||
|C3|construction||||||
|C4|natural||||||
|C4|play_field||||||
|C4|path&stair||||||
|C4|vehicle||||||
|C4|not vehicle||||||
|C4|building||||||
|C4|link||||||
|C4|facility||||||
|C5|natural||||||
|C5|play_field||||||
|C5|sheltered||||||
|C5|unsheltered||||||
|C5|bus_stop||||||
|C5|car||||||
|C5|bus||||||
|C5|not vehicle||||||
|C5|wall||||||
|C5|roof||||||
|C5|link||||||
|C5|artificial_landscape||||||
|C5|lamp||||||
|C5|others||||||
