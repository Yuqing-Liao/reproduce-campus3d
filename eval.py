from __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.optim as optim
from utils.metric import IouMetric, AccuracyMetric, HierarchicalConsistency
from utils import config, metric
from dataset.loader import TorchDataset, TorchDataLoader
from models.DGCNN.dgcnn_model import DGCNN
from models.PointNet2.pointnet2_model import PointNet2
from models.PointCNN.pointcnn_model import PointCNN
import numpy as np
from utils.io import load_model, save_model
from utils.interpolation import interpolate
from utils.label_fusion import heirarchical_ensemble


def test(args, io, cfg, HM):
    test_dataset = TorchDataset("TEST_SET", params=cfg.DATASET, is_training=False, )
    test_loader = TorchDataLoader(dataset=test_dataset,
                                  batch_size=cfg.TRAIN.BATCH_SIZE,
                                  num_workers=cfg.TRAIN.BATCH_SIZE)
    io.cprint('Eval size:{}'.format(test_dataset.labels[0].shape))
    device = torch.device("cuda" if args.cuda else "cpu")


    #Try to load models
    if args.mc_level == -1:
        if args.model == 'dgcnn':
            model = DGCNN(cfg).to(device)
        elif args.model == 'pointnet2':
            model = PointNet2(cfg, args).to(device)
        elif args.model == 'pointcnn':
            model = PointCNN(cfg).to(device)
        else:
            raise Exception("Not implemented")
        model = load_model(args, cfg, model)
    else:
        models = []
        for i in range(5):
            args.mc_level = i
            model = PointNet2(cfg, args).to(device)
            model = load_model(args, cfg, model, i)
            models.append(model)
            

    test_loss = 0.0
    total_num = 0
    count = 0
    correct = [0, 0, 0, 0, 0]
    logits_collections = [[] for _ in range(len(cfg.DATASET.DATA.LABEL_NUMBER))]
    points_collections = []
    # cfs_mtx_list = [metric.IouMetric(list(range(l))) for l in cfg.DATASET.DATA.LABEL_NUMBER]
    if args.mc_level == -1:
        model.eval()
    else:
        for i in range(5):
            models[i].eval()
    with torch.no_grad():
        for batch_idx, data_ in enumerate(test_loader):
            points_centered, labels, colors, raw_points = data_
            # (16, 2048, 3), (16, 2048, 5)
            if labels.shape[0] < cfg.TRAIN.BATCH_SIZE:
                break
            points_clrs = torch.FloatTensor(np.concatenate([points_centered, colors], axis=-1))
            points_clrs = points_clrs.to(device).permute(0, 2, 1)  # (batch_size, dim, nums_point)
            if args.mc_level==-1:
                seg_pred = model(points_clrs)
            else:
                seg_pred = []
                for i in range(5):
                    pred = models[i](points_clrs)
                    seg_pred.append(pred)
            points_collections.append(raw_points)
            for pred, collects in zip(seg_pred, logits_collections):
                collects.append(pred.cpu().detach().permute(0, 2, 1))
    points = np.concatenate(points_collections)
    logits = [np.concatenate(lgs) for lgs in logits_collections]
    logits = [lgs.reshape(lgs.shape[0]*lgs.shape[1], lgs.shape[2]) for lgs in logits]
    path_label = heirarchical_ensemble(logits, HM, weight=np.full((5,), 1.0))
    points = points.reshape(points.shape[0]*points.shape[1], points.shape[2])
    D, I = interpolate(sparse_points=points, dense_points=test_dataset.points[0])

    io.cprint('Cal IoU/OA MT')
    pred_labels = []
    for i in range(len(logits)):
        io.cprint('IoU {}'.format(i))
        tmp_label = np.argmax(logits[i], axis=1)
        new_label = tmp_label[I]
        iou = IouMetric.cal_iou(np.squeeze(new_label), test_dataset.labels[0][..., i],
                                label_range=list(range(logits[i].shape[-1])))

        iou_string = [str(layer_iou) for layer_iou in iou]
        iou_string = '\n'.join(iou_string)
        io.cprint(iou_string)
        oa = AccuracyMetric.cal_oa(pred=np.squeeze(new_label), target=test_dataset.labels[0][..., i])
        io.cprint('OA {}:{}'.format(i, oa))
        pred_labels.append(new_label)

    labels = np.asarray(pred_labels).transpose()
    cr = HierarchicalConsistency.cal_consistency_rate(HM, np.squeeze(labels))
    io.cprint('Cal consistent rate MT: {}'.format(cr))

    io.cprint('Cal IoU/OA HE')
    pred_labels = []
    for i in range(len(logits)):
        io.cprint('IoU {}'.format(i))
        new_label = path_label[..., i][I]
        iou = IouMetric.cal_iou(np.squeeze(new_label), test_dataset.labels[0][..., i],
                                label_range=list(range(logits[i].shape[-1])))
        iou_string = [str(layer_iou) for layer_iou in iou]
        iou_string = '\n'.join(iou_string)
        io.cprint(iou_string)
        oa = AccuracyMetric.cal_oa(pred=np.squeeze(new_label), target=test_dataset.labels[0][..., i])
        io.cprint('OA {}:{}'.format(i, oa))
        pred_labels.append(new_label)

    labels = np.asarray(pred_labels).transpose()
    cr = HierarchicalConsistency.cal_consistency_rate(HM, np.squeeze(labels))
    io.cprint('Cal consistent rate HE: {}'.format(cr))
