# Campus3D
## Introduction
## Installation
## Training
### Train from scratch
Check the configuration files in `config/` and run experiments, eg:
```
python run.py --model 'pointnet2' --mc_level -1 --exp_name 'pointnet2_MT'
```
### Train from pretrained model
## Evaluation
## MODEL ZOO
### Models
|Model|Method|MC Level|Training Process|Scheduler|Dropout<br>Rate|Download<br>Link|
:-:|:-:|:-:|:-:|:-:|:-:|:-:
|Pointnet++|MC|0|50 epochs(lr=0.01)|cos|0.5|[MC0]()|
|Pointne++|MC|1|50 epochs(lr=0.01)|cos|0.5|[MC1]()|
|Pointnet++|MC|2|50 epochs(lr=0.01)|cos|0.5|[MC2]()|
|Pointnet++|MC|3|50 epochs(lr=0.01)|cos|0.5|[MC3]()|
|Pointnet++|MC|4|50 epochs(lr=0.01)|cos|0.5|[MC4]()|
|Pointnet++|MTnc|-1|50 epochs(lr=0.01)|cos|0.5|[pointnet2_MTnc]()|
|PointCNN|MTnc|-1|50 epochs(lr=0.01)|cos|0.5|[pointcnn_MTnc]()|
|DGCNN|MTnc|-1|50 epochs(lr=0.01)|cos|0.5|[dgcnn_MTnc]()|
|Pointnet++|MT|-1|50 epochs(lr=0.01) +<br>20 epochs(lr=0.01)|cos|0.5|[pointnet2_MT]()|
|PointCNN|MT|-1|50 epochs(lr=0.01) +<br>30 epochs(lr=0.01)|cos|0.5|[pointcnn_MT]()|
|DGCNN|MT|-1|50 epochs(lr=0.01) +<br>20 epochs(lr=0.01)|cos|0.5|[dgcnn_MT]()|

### Semantic segmentation benchmarks for model PointNet2 with different methods <br>
|Granularity Level|Class|MC|MC+HE|MTnc|MT|MT+HE|
:-:|:-:|:-:|:-:|:-:|:-:|:-:
|C1|ground|85.4|86.4|85.3|86.1|86.1|
|C1|construction|79.9|80.8|79.4|81.4|81.5|
|C2|natural|81.1|82.4|80.8|82.9|82.9|
|C2|man_made|58.5|60.9|58.7|58.1|58.5|
|C2|construction|78.8|80.8|78.5|81.3|81.5|
|C3|natural|79.2|82.4|80.8|82.9|82.9|
|C3|play_field|62.9|65.9|56.1|66.5|67.3|
|C3|path&stair|8.7|8.2|8.7|0.0|0.0|
|C3|driving_road|58.4|60.6|57.7|58.4|58.5|
|C3|construction|76.6|80.8|78.2|81.4|81.5|
|C4|natural|81.0|82.4|80.4|82.9|82.9|
|C4|play_field|57.3|65.9|54.0|68.2|67.3|
|C4|path&stair|9.3|8.2|8.7|0.0|0.0|
|C4|vehicle|16.6|19.4|16.7|9.4|9.9|
|C4|not vehicle|57.9|59.9|57.2|57.8|57.9|
|C4|building|76.5|78.2|75.4|78.8|78.8|
|C4|link|0.0|0.1|0.2|0.0|0.0|
|C4|facility|0.0|0.0|0.0|0.0|0.0|
|C5|natural|80.1|82.4|80.5|83.0|82.9|
|C5|play_field|52.7|65.9|53.5|67.0|67.3|
|C5|sheltered|10.6|7.9|9.0|0.0|0.0|
|C5|unsheltered|7.6|7.9|8.3|0.0|0.0|
|C5|bus_stop|0.0|0.0|0.1|0.0|0.0|
|C5|car|18.1|19.5|16.7|10.6|9.9|
|C5|bus|0.0|0.0|0.0|0.0|0.0|
|C5|not vehicle|58.1|59.9|57.4|57.7|57.9|
|C5|wall|46.5|47.3|45.8|47.3|47.1|
|C5|roof|43.1|44.2|41.7|47.4|47.4|
|C5|link|0.2|0.1|0.4|0.0|0.0|
|C5|artificial_landscape|0.0|0.0|0.0|0.0|0.0|
|C5|lamp|0.0|0.0|0.0|0.0|0.0|
|C5|others|0.0|0.0|0.0|0.0|0.0|

### Semantic segmentation benchmarks(mIoU% and OA%) for three feature learning models with MT+HE <br>
|Benchmark|Model|C1|C2|C3|C4|C5|
:-:|:-:|:-:|:-:|:-:|:-:|:-:
|OA%|PointNet++|91.4|87.5|86.7|85.0|75.1|
|OA%|PointCNN|88.9|79.3|78.7|76.8|63.8|
|OA%|DGCNN|94.7|90.6|89.1|87.2|81.5|
|mIoU%|PointNet++|83.8|74.3|58.0|37.1|22.3|
|mIoU%|PointCNN|79.7|61.5|42.8|26.3|15.0|
|mIoU%|DGCNN|89.6|80.1|63.3|43.1|28.4|
