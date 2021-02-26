from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from utils import config, metric
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from dataset.loader import TorchDataset, TorchDataLoader
from models.DGCNN.dgcnn_model import DGCNN
from models.PointNet2.pointnet2_model import PointNet2
from models.PointCNN.pointcnn_model import PointCNN
from dataset.reader import read_h_matrix_file_list
from eval import test
import numpy as np
from utils.io import IOStream, load_model, save_model
from utils.loss import ConsistencyLoss, HeirarchicalCrossEntropyLoss


def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('checkpoints/'+args.exp_name+'/'+'models')
    os.system('cp run.py checkpoints'+'/'+args.exp_name+'/'+'run.py.backup')
    os.system('cp configs/train_setting.yaml checkpoints/' + args.exp_name + '/train_setting.backup')


def full_batch_size(batch_size, *np_args):
    sample_size = np_args[0].shape[0]
    init_ind = np.arange(sample_size)
    if sample_size < batch_size:
        res_ind = np.random.randint(0, sample_size, (batch_size - sample_size, ))
        np_args = [np.concatenate([arr, arr[res_ind]]) for arr in np_args]
    return tuple([init_ind] + list(np_args))


def cal_correct(pred, target):
    return torch.eq(target.squeeze(), pred.argmax(dim=2)).sum().item()


def train(args, io, cfg, HM):
    CM = [HM[i + 1, i] for i in range(len(HM.classes_num) - 1)]
    CLW = cfg.TRAIN.CONSISTENCY_WEIGHTS
    l = args.mc_level
    enable_consistency_loss = cfg.TRAIN.CONSISTENCY_LOSS
    device = torch.device("cuda" if args.cuda else "cpu")
    if enable_consistency_loss and l == -1:
        ConsistencyLossCal = ConsistencyLoss(CM, CLW, device)
    max_epoch = cfg.TRAIN.MAX_EPOCH
    
    #Try to load models
    if args.model == 'dgcnn':
        model = DGCNN(cfg).to(device)
    elif args.model == 'pointnet2':
        model = PointNet2(cfg, args).to(device)
    elif args.model == 'pointcnn':
        model = PointCNN(cfg).to(device)
    else:
        raise Exception("Not implemented")
    if cfg.TRAIN.IS_PRETRAINED:
        model = load_model(args, cfg, model)
        
    train_dataset = TorchDataset("TRAIN_SET", params=cfg.DATASET, is_training=True, )
    train_loader = TorchDataLoader(dataset=train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
                                   num_workers=4)
                                   
    validation_dataset = TorchDataset("VALIDATION_SET", params=cfg.DATASET,
                                      is_training=True, )
    validation_loader = TorchDataLoader(dataset=validation_dataset,
                                        batch_size=cfg.TRAIN.BATCH_SIZE,
                                        num_workers=4)
    io.cprint('length of train loader: %d' % (len(train_loader)))

    HCrossEntropy = HeirarchicalCrossEntropyLoss(train_dataset.data_sampler.label_weights,device)
    
    opt = optim.SGD(model.parameters(), lr=cfg.TRAIN.LEARNING_RATE, momentum=cfg.TRAIN.MOMENTUM, weight_decay=1e-4)

    if cfg.TRAIN.SCHEDULER == 'cos':
        scheduler = CosineAnnealingLR(opt, max_epoch, eta_min=1e-3)
    elif cfg.TRAIN.SCHEDULER == 'step':
        scheduler = StepLR(opt, 20, 0.5)

    for epoch in range(max_epoch):
        ####################
        # Train
        ####################
        io.cprint('___________________epoch %d_____________________' %(epoch))
        train_loss = 0.0
        total_num = 0
        count = 0
        cfs_mtx_list = [metric.IouMetric(list(range(l))) for l in cfg.DATASET.DATA.LABEL_NUMBER]
        model.train()
        for batch_idx, data_ in enumerate(train_loader):
            points_centered, labels, colors, label_weights = data_
            if labels.shape[0] < cfg.TRAIN.BATCH_SIZE:
                break
            points_clrs = torch.FloatTensor(np.concatenate([points_centered, colors], axis=-1))
            points_clrs = points_clrs.to(device).permute(0, 2, 1) # (batch_size, dim, nums_point)
            labels = torch.LongTensor(labels).to(device)
            label_weights = torch.Tensor(label_weights).to(device)
            num_points = labels.size()[1]
            batch_size = labels.size()[0]
            opt.zero_grad()
            seg_pred = model(points_clrs)
            MTLoss = 0.
            labels_np = labels.cpu().detach().numpy()
            labels = torch.chunk(labels, 5, dim=2)
            label_weights = torch.chunk(label_weights, 5, dim=2)
            level_weights = cfg.TRAIN.LOSS_WEIGHTS
            if l == -1:
                for i in range(len(seg_pred)):
                    seg_pred_i = seg_pred[i].permute(0, 2, 1).contiguous() #(batch_size, num_points, cls)
                    MTLoss += HCrossEntropy(seg_pred_i, labels[i], level=i) * level_weights[i]
            else:
                seg_pred = seg_pred.permute(0, 2, 1).contiguous()
                MTLoss += HCrossEntropy(seg_pred, labels[l], level=l)
                pred_np = np.argmax(seg_pred.cpu().detach().numpy(), 2)      
            if enable_consistency_loss and l == -1:
                CLoss = ConsistencyLossCal(seg_pred)
                MTLoss += CLoss
            MTLoss.backward()
            opt.step()
            count += batch_size
            train_loss += MTLoss.item()

            if batch_idx != 0 and batch_idx % 500 == 0:
                io.cprint('batch: %d, _loss: %f' %(batch_idx, MTLoss))
 
        if cfg.TRAIN.SCHEDULER == 'cos':
            scheduler.step()
        elif cfg.TRAIN.SCHEDULER == 'step':
            if opt.param_groups[0]['lr'] > 1e-5:
                scheduler.step()
            if opt.param_groups[0]['lr'] < 1e-5:
                for param_group in opt.param_groups:
                    param_group['lr'] = 1e-5
        io.cprint('train %d, loss: %f' % (epoch, train_loss*1.0/count))

        ####################
        # Test(validation)
        ####################
        if epoch % 3 == 0:
            cfs_mtx_list = [metric.IouMetric(list(range(l))) for l in cfg.DATASET.DATA.LABEL_NUMBER]
            model.eval()
            all_correct = torch.Tensor([0, 0, 0, 0, 0])
            all_heads_label = [[] for _ in range(len(HM.classes_num))]
            with torch.no_grad():
                for batch_idx, data_ in enumerate(validation_loader):
                    points_centered, labels, colors, label_weights = data_
                    if labels.shape[0] < cfg.TRAIN.BATCH_SIZE:
                        break
                    points_clrs = torch.FloatTensor(np.concatenate([points_centered, colors], axis=-1))
                    points_clrs = points_clrs.to(device).permute(0, 2, 1)  # (batch_size, dim, nums_point)
                    labels = torch.LongTensor(labels).to(device)
                    label_weights = torch.Tensor(label_weights).to(device)
                    num_points = labels.size()[1]
                    batch_size = labels.size()[0]
                    labels_np = labels.cpu().detach().numpy()
                    labels = torch.chunk(labels, 5, dim=2)
                    opt.zero_grad()
                    seg_pred = model(points_clrs)
                    total_num += num_points*batch_size
                    if l == -1:
                        for i in range(len(seg_pred)):
                            seg_pred_i = seg_pred[i].permute(0, 2, 1).contiguous() #(batch_size, num_points, cls)
                            all_correct[i] += cal_correct(seg_pred_i, labels[i])
                            pred_np = np.argmax(seg_pred_i.cpu().detach().numpy(), 2)
                            cfs_mtx_list[i].update(pred_np, labels_np[..., i])
                            all_heads_label[i].append(pred_np.reshape(-1))
                    else:
                        seg_pred = seg_pred.permute(0, 2, 1).contiguous()
                        all_correct[l] += cal_correct(seg_pred, labels[l])
                        pred_np = np.argmax(seg_pred.cpu().detach().numpy(), 2)
                        cfs_mtx_list[l].update(pred_np, labels_np[..., l])
                if l == -1:
                    all_heads_label = np.asarray([np.concatenate(l) for l in all_heads_label]).transpose()
                    scores = metric.HierarchicalConsistency.cal_consistency_rate(HM, all_heads_label)
                    io.cprint('consistency score: {}'.format(scores))
            io.cprint('test aver acc: {}'.format({i: crt*1.0/total_num for i, crt in enumerate(all_correct)}))
            io.cprint('eval avg class IoU: {}'.format('\n'.join([str(m.avg_iou()) for m in cfs_mtx_list])))


        if epoch % 5 == 0:
            save_model(model, cfg, args, 'model')
    save_model(model, cfg, args, 'model_final')


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Part Segmentation')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='pointnet2', metavar='N',
                        choices=['dgcnn', 'pointnet2', 'pointcnn'],
                        help='Model to use, [dgcnn, pointnet2, pointcnn]')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')   
    parser.add_argument('--mc_level', type=int, default=-1, help='label level to use; -1 means all')      
    args = parser.parse_args()

    abs_cfg_dir = os.path.abspath(os.path.join(__file__, "../configs"))
    config.merge_cfg_from_dir(abs_cfg_dir)
    cfg = config.CONFIG
    
    HM = read_h_matrix_file_list(cfg.DATASET.DATA.H_MATRIX_LIST_FILE)
    _init_()
    name_dict = {True:"eval", False:""}
    io = IOStream('checkpoints/' + args.exp_name + '/{}run.log'.format(name_dict[args.eval]))

    args.cuda = torch.cuda.is_available()
    torch.manual_seed(cfg.DEVICES.SEED)

    if args.cuda:
        if len(cfg.DEVICES.GPU_ID) == 1:
            torch.cuda.set_device(cfg.DEVICES.GPU_ID[0])
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(cfg.DEVICES.SEED)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io, cfg, HM)
    else:
        test(args, io, cfg, HM)
