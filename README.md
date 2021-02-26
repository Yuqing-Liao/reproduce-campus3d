# Campus3D

## Introduction
The repository contains the re-implementation of this [ACM MM 2020 Paper](https://3d.dataset.site) based on the [repository](https://github.com/shinke-li/Campus3D/). It also presents the reproduced results of the supported paper with trained models in **MODEL ZOO**. The reduced version of Campus3D dataset can be donwloaded from the [official website](https://3d.dataset.site) or the [alternative](https://3d.nus.app). 

## Installation
The whole package can be downloaded by the following command.
```
git clone https://github.com/Yuqing-Liao/reproduce-campus3d.git
```
Dependencies can be installed using the provided script.
```
cd reproduce-campus3d
pip install -r requirements.txt
```
Compressed Campus3D dataset file `campus3d-reduce.zip` can be downloaded from [official website](https://3d.dataset.site). Put it into `data/` and unzip with below script
```
cd reproduce-campus3d/data
unzip campus3d-reduce.zip
```

## Training & Evaluation
### Train from scratch
To apply training of the model, please first check the configuration files in `config/`. Particularly you need to change the value of `IS_PRETRAINED` to false and then run experiments, eg:
```
cd reproduce-campus3d
python run.py --model 'pointnet2' --mc_level -1 --exp_name 'EXP_NAME'
```
The 'EXP_NAME' is the user-defined name. In this way, the models will be saved in `checkpoints/EXP_NAME/models`, and other output files will be saved in `checkpoints/EXP_NAME`.

### Train from pretrained model
Pretrained models are available on Google Drive, and they can be downloaded through the link presented in the following table. You can train either from the downloaded models or from your own pretrained models. To apply training of the model, please first check the configuration files in `config/`. Particularly you need to change the value of `IS_PRETRAINED` to false, `PRETRAINED_MODEL_PATH` to the path of the model to train and then run experiments, eg:
```
cd reproduce-campus3d
python run.py --model 'pointnet2' --mc_level -1 --exp_name 'EXP_NAME'
```
In this way, the models will be saved in `checkpoints/EXP_NAME/models`, and other output files will be saved in `checkpoints/EXP_NAME`.

### Evaluation
To apply evaluation of the model on the test set, please first check the configuration files in `config/`. Particularly you need to change the value of `PRETRIANED_MODEL_PATH` to the path of the model to evaluate and then run experiments, eg:
```
cd reproduce-campus3d
python run.py --eval true --model 'pointnet2' --mc_level -1 --exp_name 'EXP_NAME'
```
In this way, the output files will be saved in `check/EXP_NAME`.

## Experiments
### Hierarchical Learning (HL) Experiments
The hierarchical learning experiments were proposed to present the effectiveness of the **Multi-task and Hierarchical Esemble(MT+HE)** method. **Multi-classifiers(MC)** in each level were also proposed for comparison. To run the training, the argument `--mc_level` can be set as **0-4** and **-1** for MC experiments in 0-4 levels and MT+HE experiments in all levels respectively. In addition, the MT training contains two stage **Multi-task Learning without consistency loss(MT<sub>nc</sub>)** and **Multi-task Learbing with consistency loss(MT)**, of which the MT is trained based on the pretrained MT<sub>nc</sub> model. To run the evaluation, 

### Benchmark Experiments
The semantic segmentation bechmark were built with three models PointNet++, PointCNN and DGCNN. They are all conducted via the MT+HE method for hierarchical learning on the Campus3D dataset. To run different models, one can change the argument `--model` as the indicated model. Following are the reference repository for PyTorch implementation of 3D deep models.

PointNet++ [GitHub Link](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)

PointCNN [GitHub Link](https://github.com/agarret7/PointCNN)

DGCNN [GitHub Link](https://github.com/WangYueFt/dgcnn)

## MODEL ZOO
### Models
｜No.|Model|Name|Method|MC Level|Training Process|Scheduler|Download<br>Link|
:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:
0|PointNet++|'pointnet2'|MC|0|50 epochs(lr=0.01)|cos|[MC0](https://drive.google.com/file/d/1XrCGYfFwFBx6y4O6CY71YFGXFxCUwSbc/view?usp=sharing)|
1|PointNe++|'pointnet2'|MC|1|50 epochs(lr=0.01)|cos|[MC1](https://drive.google.com/file/d/1oUOyuszPHDjZsRDhdJGwIHXxImllvvRT/view?usp=sharing)|
2|PointNet++|'pointnet2'|MC|2|50 epochs(lr=0.01)|cos|[MC2](https://drive.google.com/file/d/1peZa8j-HMI4-NCfnSSjpYXrHmI1M3AG1/view?usp=sharing)|
3|PointNet++|'pointnet2'|MC|3|50 epochs(lr=0.01)|cos|[MC3](https://drive.google.com/file/d/1aXPemqQCXWL33DdlMB86jnhNCx7-hoNB/view?usp=sharing)|
4|PointNet++|'pointnet2'|MC|4|50 epochs(lr=0.01)|cos|[MC4](https://drive.google.com/file/d/1ilJXKI42pcbenK7Q2LRWzOAm4ui6teWI/view?usp=sharing)|
5|PointNet++|'pointnet2'|MT<sub>nc</sub>|-1|50 epochs (lr=0.01)|cos|[pointnet2_MTnc](https://drive.google.com/file/d/1QducufhXMk65LO5ZNJx-1kLRg42yf2L6/view?usp=sharing)|
6|PointCNN|'pointcnn'|MT<sub>nc</sub>|-1|50 epochs (lr=0.01)|cos|[pointcnn_MTnc](https://drive.google.com/file/d/1NAaNVMtq79AyxYhS8Caz2LpTyxl0tntw/view?usp=sharing)|
7|DGCNN|'dgcnn'|MT<sub>nc</sub>|-1|50 epochs (lr=0.01)|cos|[dgcnn_MTnc](https://drive.google.com/file/d/1-CQHSkdda30j7Zq9HyC0ZITGsdiHsZ0W/view?usp=sharing)|
8|Pointnet++|'pointnet2'|MT|-1|50 epochs (lr=0.01) +<br>20 epochs with<br>consistency loss (lr=0.01)|cos|[pointnet2_MT](https://drive.google.com/file/d/1eY1WZ9JYjXUrCPqLU6UegC_F3pojzAls/view?usp=sharing)|
9|PointCNN|'pointcnn'|MT|-1|50 epochs (lr=0.01) +<br>30 epochs with<br>consistency loss (lr=0.01)|cos|[pointcnn_MT](https://drive.google.com/file/d/1l9kda3z5359aI08ZpdRDJRm6YItvv_3N/view?usp=sharing)|
10|DGCNN|'dgcnn'|MT|-1|50 epochs (lr=0.01) +<br>20 epochs with<br>consistency loss (lr=0.01)|cos|[dgcnn_MT](https://drive.google.com/file/d/1qo157dARwZhZ5R_AUDSbs_bE_T5S0bD-/view?usp=sharing)|

### Benchmark Experiments Results
#### Semantic segmentation benchmarks (mIoU% and OA%) for models with MT+HE method
|Benchmark|Model|C<sup>1</sup>|C<sup>2</sup>|C<sup>3</sup>|C<sup>4</sup>|C<sup>5</sup>|
:-:|:-:|:-:|:-:|:-:|:-:|:-:
|OA%|PointNet++|91.4|87.5|86.7|85.0|75.1|
|OA%|PointCNN|88.9|79.3|78.7|76.8|63.8|
|OA%|DGCNN|94.7|90.6|89.1|87.2|81.5|
|mIoU%|PointNet++|83.8|74.3|58.0|37.1|22.3|
|mIoU%|PointCNN|79.7|61.5|42.8|26.3|15.0|
|mIoU%|DGCNN|89.6|80.1|63.3|43.1|28.4|
These results are produced by model No.8, No.9 and No.10.

### Hierarchical Learning Experiments Results
#### Semantic segmentation benchmarks(OA%) for different HL methods with model PointNet++
|Method|C<sup>1</sup>|C<sup>2</sup>|C<sup>3</sup>|C<sup>4</sup>|C<sup>5</sup>|
:-:|:-:|:-:|:-:|:-:|:-:
|MC|90.8|86.2|84.4|83.6|73.6|
|MC+HE|91.4|87.4|86.5|84.8|74.9|
|MT<sub>nc</sub>|90.6|86.0|85.0|83.1|73.3|
|MT|91.4|87.4|86.7|84.9|75.2|
|MT+HE|91.4|87.5|86.7|85.0|75.1|
These results are produced by model No.0-4, No.5 and No.8. They demonstrate the effectiveness of the MT+HE method fro HL problem.
Results with detailed per-class mIoU are displayed below. 

#### Semantic segmentation benchmarks(IoU%) for different HL methods with model PointNet++
|Granularity Level|Class|MC|MC+HE|MTnc|MT|MT+HE|
:-:|:-:|:-:|:-:|:-:|:-:|:-:
|C<sup>1</sup>|ground|85.4|86.4|85.3|86.1|86.1|
|C<sup>1</sup>|construction|79.9|80.8|79.4|81.4|81.5|
|C<sup>2</sup>|natural|81.1|82.4|80.8|82.9|82.9|
|C<sup>2</sup>|man_made|58.5|60.9|58.7|58.1|58.5|
|C<sup>2</sup>|construction|78.8|80.8|78.5|81.3|81.5|
|C<sup>3</sup>|natural|79.2|82.4|80.8|82.9|82.9|
|C<sup>3</sup>|play_field|62.9|65.9|56.1|66.5|67.3|
|C<sup>3</sup>|path&stair|8.7|8.2|8.7|0.0|0.0|
|C<sup>3</sup>|driving_road|58.4|60.6|57.7|58.4|58.5|
|C<sup>3</sup>|construction|76.6|80.8|78.2|81.4|81.5|
|C<sup>4</sup>|natural|81.0|82.4|80.4|82.9|82.9|
|C<sup>4</sup>|play_field|57.3|65.9|54.0|68.2|67.3|
|C<sup>4</sup>|path&stair|9.3|8.2|8.7|0.0|0.0|
|C<sup>4</sup>|vehicle|16.6|19.4|16.7|9.4|9.9|
|C<sup>4</sup>|not vehicle|57.9|59.9|57.2|57.8|57.9|
|C<sup>4</sup>|building|76.5|78.2|75.4|78.8|78.8|
|C<sup>4</sup>|link|0.0|0.1|0.2|0.0|0.0|
|C<sup>4</sup>|facility|0.0|0.0|0.0|0.0|0.0|
|C<sup>5</sup>|natural|80.1|82.4|80.5|83.0|82.9|
|C<sup>5</sup>|play_field|52.7|65.9|53.5|67.0|67.3|
|C<sup>5</sup>|sheltered|10.6|7.9|9.0|0.0|0.0|
|C<sup>5</sup>|unsheltered|7.6|7.9|8.3|0.0|0.0|
|C<sup>5</sup>|bus_stop|0.0|0.0|0.1|0.0|0.0|
|C<sup>5</sup>|car|18.1|19.5|16.7|10.6|9.9|
|C<sup>5</sup>|bus|0.0|0.0|0.0|0.0|0.0|
|C<sup>5</sup>|not vehicle|58.1|59.9|57.4|57.7|57.9|
|C<sup>5</sup>|wall|46.5|47.3|45.8|47.3|47.1|
|C<sup>5</sup>|roof|43.1|44.2|41.7|47.4|47.4|
|C<sup>5</sup>|link|0.2|0.1|0.4|0.0|0.0|
|C<sup>5</sup>|artificial_landscape|0.0|0.0|0.0|0.0|0.0|
|C<sup>5</sup>|lamp|0.0|0.0|0.0|0.0|0.0|
|C<sup>5</sup>|others|0.0|0.0|0.0|0.0|0.0|

