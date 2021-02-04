import torch
import torch.nn as nn
import torch.nn.functional as F
from .pointnet_util import PointNetSetAbstraction, PointNetFeaturePropagation

class PointNet2(nn.Module):
    def __init__(self, cfg=None):
        super(PointNet2, self).__init__()
        self.cfg = cfg
        self.num_class = cfg.DATASET.DATA.LABEL_NUMBER
        self.level = cfg.TRAIN.MC_LEVEL

        self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, 6 + 3, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 128 + 3, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256 + 3, [256, 256, 512], False)
        
        self.decoder0 = PointNet2Decoder(self.num_class[0], self.cfg)
        self.decoder1 = PointNet2Decoder(self.num_class[1], self.cfg)
        self.decoder2 = PointNet2Decoder(self.num_class[2], self.cfg)
        self.decoder3 = PointNet2Decoder(self.num_class[3], self.cfg)
        self.decoder4 = PointNet2Decoder(self.num_class[4], self.cfg)

    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:, :3, :]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        input_list = [l0_xyz, l0_points,
                      l1_xyz, l1_points,
                      l2_xyz, l2_points,
                      l3_xyz, l3_points,
                      l4_xyz, l4_points]
        x0 = self.decoder0(*input_list)
        x1 = self.decoder1(*input_list)
        x2 = self.decoder2(*input_list)
        x3 = self.decoder3(*input_list)
        x4 = self.decoder4(*input_list)
        return [x0, x1, x2, x3, x4]
            
class PointNet2Decoder(nn.Module):
    def __init__(self, num_class, cfg):
        super(PointNet2Decoder, self).__init__()
        self.cfg = cfg
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(cfg.TRAIN.DROPOUT_RATE)
        self.conv_out = nn.Conv1d(128, num_class, 1)

    def forward(self,   l0_xyz, l0_points,
                        l1_xyz, l1_points,
                        l2_xyz, l2_points,
                        l3_xyz, l3_points,
                        l4_xyz, l4_points):
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv_out(x)
        return x  
    

class PointNet2_MC(nn.Module):
    def __init__(self, cfg=None):
        super(PointNet2_MC, self).__init__()
        self.cfg = cfg
        self.num_class = cfg.DATASET.DATA.LABEL_NUMBER
        self.level = cfg.TRAIN.MC_LEVEL

        self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, 6 + 3, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 128 + 3, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256 + 3, [256, 256, 512], False)
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2_0 = nn.Conv1d(128, self.num_class[0], 1)
        self.conv2_1 = nn.Conv1d(128, self.num_class[1], 1)
        self.conv2_2 = nn.Conv1d(128, self.num_class[2], 1)
        self.conv2_3 = nn.Conv1d(128, self.num_class[3], 1)
        self.conv2_4 = nn.Conv1d(128, self.num_class[4], 1)

    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:, :3, :]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x0 = self.conv2_0(x)
        x1 = self.conv2_1(x)
        x2 = self.conv2_2(x)
        x3 = self.conv2_3(x)
        x4 = self.conv2_4(x)
        if self.level == 0:
            return x0
        elif self.level == 1:
            return x1
        elif self.level == 2:
            return x2
        elif self.level == 3:
            return x3
        elif self.level == 4:
            return x4

if __name__=='__main__':
    device = torch.device("cuda")
    model = PointNet2().to(device)
    rand_input = torch.rand(32, 6, 2048).to(device)
    output = model(rand_input)
    print(output)
        