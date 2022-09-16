import argparse
import time
import os
import gc
import numpy as np
from collections import OrderedDict

import torch.nn.functional as F

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from lib.network.rtpose import get_model as get_student_model, use_mobilenet
from lib.network.rtpose_vgg import get_model as get_teacher_model
from lib.network.rtpose_mobilenetV33 import mobilenetv3
from lib.datasets import transforms, datasets

from tqdm import tqdm
from tensorboardX import SummaryWriter

#DATA_DIR = '/data/coco'
DATA_DIR = 'lib/datasets/coco'

if torch.backends.cudnn.is_available():
    torch.backends.cudnn.enabled = torch.backends.cudnn.benchmark = True

ANNOTATIONS_TRAIN = [os.path.join(DATA_DIR, 'annotations', item) for item in ['person_keypoints_train2017.json']]
ANNOTATIONS_VAL = os.path.join(DATA_DIR, 'annotations', 'person_keypoints_val2017.json')
IMAGE_DIR_TRAIN = os.path.join(DATA_DIR, 'images/train2017')
IMAGE_DIR_VAL = os.path.join(DATA_DIR, 'images/val2017')

def build_names():
    names = []

    for j in range(1, 7):
        for k in range(1, 3):
            names.append('loss_stage%d_L%d' % (j, k))
    return names

def train_cli(parser):
    group = parser.add_argument_group('dataset and loader')
    group.add_argument('--train-annotations', default=ANNOTATIONS_TRAIN)
    group.add_argument('--train-image-dir', default=IMAGE_DIR_TRAIN)
    group.add_argument('--val-annotations', default=ANNOTATIONS_VAL)
    group.add_argument('--val-image-dir', default=IMAGE_DIR_VAL)
    group.add_argument('--pre-n-images', default=8000, type=int,
                       help='number of images to sampe for pretraining')
    group.add_argument('--n-images', default=None, type=int,
                       help='number of images to sample')
    group.add_argument('--duplicate-data', default=None, type=int,
                       help='duplicate data')
    group.add_argument('--loader-workers', default=8, type=int,
                       help='number of workers for data loading')
    group.add_argument('--batch-size', default=72, type=int,
                       help='batch size')
    group.add_argument('--lr', '--learning-rate', default=1, type=float,
                    metavar='LR', help='initial learning rate')
    group.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
    group.add_argument('--weight-decay', '--wd', default=0.000, type=float,
                    metavar='W', help='weight decay (default: 1e-4)') 
    group.add_argument('--nesterov', dest='nesterov', default=True, type=bool)     
    group.add_argument('--print_freq', default=20, type=int, metavar='N',
                    help='number of iterations to print the training statistics')    
                   
                                         
def train_factory(args, preprocess, target_transforms):
    train_datas = [datasets.CocoKeypoints(
        root=args.train_image_dir,
        annFile=item,
        preprocess=preprocess,
        image_transform=transforms.image_transform_train,
        target_transforms=target_transforms,
        n_images=args.n_images,
    ) for item in args.train_annotations]

    train_data = torch.utils.data.ConcatDataset(train_datas)
    
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        pin_memory=args.pin_memory, num_workers=args.loader_workers, drop_last=True)

    val_data = datasets.CocoKeypoints(
        root=args.val_image_dir,
        annFile=args.val_annotations,
        preprocess=preprocess,
        image_transform=transforms.image_transform_train,
        target_transforms=target_transforms,
        n_images=args.n_images,
    )
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False,
        pin_memory=args.pin_memory, num_workers=args.loader_workers, drop_last=True)

    return train_loader, val_loader, train_data, val_data

def cli():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    train_cli(parser)
    parser.add_argument('-o', '--output', default=None,
                        help='output file')
    parser.add_argument('--stride-apply', default=1, type=int,
                        help='apply and reset gradients every n batches')
    parser.add_argument('--epochs', default=75, type=int,
                        help='number of epochs to train')
    parser.add_argument('--freeze-base', default=0, type=int,
                        help='number of epochs to train with frozen base')
    parser.add_argument('--pre-lr', type=float, default=1e-4,
                        help='pre learning rate')
    parser.add_argument('--update-batchnorm-runningstatistics',
                        default=False, action='store_true',
                        help='update batch norm running statistics')
    parser.add_argument('--square-edge', default=368, type=int,
                        help='square edge of input images')
    parser.add_argument('--ema', default=1e-3, type=float,
                        help='ema decay constant')
    parser.add_argument('--debug-without-plots', default=False, action='store_true',
                        help='enable debug but dont plot')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='disable CUDA')                        
    parser.add_argument('--model_path', default='lib/network/weight/', type=str, metavar='DIR',
                    help='path to where the model saved')                         
    parser.add_argument('--teacher_weight_path', default='.', type=str, metavar='DIR',
                    help='path to where the teacher model weight path')
    args = parser.parse_args()

    # add args.device
    args.device = torch.device('cpu')
    args.pin_memory = False
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        args.pin_memory = True
        
    print(args.device)
    return args

args = cli()

writer = SummaryWriter(log_dir='./logs/')
print("Loading dataset...")
# load train data
preprocess = transforms.Compose([
        transforms.Normalize(),
        transforms.RandomApply(transforms.HFlip(), 0.5),
        transforms.RescaleRelative(),
        transforms.Crop(args.square_edge),
        transforms.CenterPad(args.square_edge),
    ])
train_loader, val_loader, train_data, val_data = train_factory(args, preprocess, target_transforms=None)

def distillation(student_result, ground_truth, teacher_result, T, alpha):
    # distillation loss + classification loss
    # y: student
    # labels: hard label
    # teacher_scores: soft label
    return nn.KLDivLoss()(F.log_softmax(student_result/T), F.softmax(teacher_result/T)) * (T*T * 2.0 + alpha) + F.cross_entropy(student_result, ground_truth) * (1.-alpha)

def get_loss(saved_for_loss, heat_temp, vec_temp):

    criterion = nn.MSELoss(reduction='mean').cuda()
    total_loss = 0

    for j in range(6):
        pred1 = saved_for_loss[2 * j]
        pred2 = saved_for_loss[2 * j + 1] 

        # Compute losses
        loss1 = criterion(pred1, vec_temp)
        loss2 = criterion(pred2, heat_temp) 

        total_loss += loss1
        total_loss += loss2
        # print(total_loss)

    return total_loss

def get_distillation_loss(saved_for_loss_student, heatmap_target, paf_target, teacher_heatmap, teacher_paf):
    distillation_loss = 0
    criterion = distillation
    #criterion = DataParallelCriterion(criterion)

    for j in range(6):
        pred1 = saved_for_loss_student[2 * j]
        pred2 = saved_for_loss_student[2 * j + 1]

        # Compute losses
        loss1 = distillation(pred1, paf_target, teacher_paf, T=20.0, alpha=0.7)
        loss2 = distillation(pred2, heatmap_target, teacher_heatmap, T=20.0, alpha=0.7)

        total_loss = get_loss(saved_for_loss_student, heatmap_target, paf_target)

        distillation_loss += loss1
        distillation_loss += loss2
        # print(total_loss)

    return distillation_loss, total_loss

def train(train_loader, student, teacher, optimizer, epoch):
    losses = AverageMeter()
    
    # switch to train mode
    student.train()
    teacher.eval()

    for img, heatmap_target, paf_target in tqdm(train_loader):

        img = img.cuda()
        heatmap_target = heatmap_target.cuda()
        paf_target = paf_target.cuda()
        # compute output
        _, saved_for_loss_student = student(img)
        (teacher_paf, teacher_heatmap), _ = teacher(img)


        distillation_loss, student_gt_loss = get_distillation_loss(saved_for_loss_student, heatmap_target, paf_target, teacher_heatmap, teacher_paf)
        losses.update(student_gt_loss, img.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        distillation_loss.backward()
        optimizer.step()

        del img, heatmap_target, paf_target, teacher_paf, teacher_heatmap, distillation_loss, student_gt_loss, saved_for_loss_student, _
        torch.cuda.empty_cache()
        gc.collect()

    return losses.avg  
        
@torch.no_grad()        
def validate(val_loader, student, teacher, epoch):
    losses = AverageMeter()
    
    student.eval()
    teacher.eval()

    for img, heatmap_target, paf_target in tqdm(val_loader):
        img = img.cuda()
        heatmap_target = heatmap_target.cuda()
        paf_target = paf_target.cuda()
        
        # compute output
        _, saved_for_loss_student = student(img)
        (teacher_paf, teacher_heatmap), _ = teacher(img)

        _, student_gt_loss = get_distillation_loss(saved_for_loss_student, heatmap_target, paf_target, teacher_heatmap, teacher_paf)

        losses.update(student_gt_loss.item(), img.size(0))
                
        del img, heatmap_target, paf_target, teacher_paf, teacher_heatmap, distillation_loss, student_gt_loss, saved_for_loss_student, _
        torch.cuda.empty_cache()
        gc.collect()

    return losses.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# model
student = get_student_model(trunk='mobilenet')
student = torch.nn.DataParallel(student).cuda()

teacher = get_teacher_model(trunk='vgg19')
teacher = torch.nn.DataParallel(teacher).cuda()
weight = torch.load(args.teacher_weight_path)
weight = {'module.'+k: v for k, v in weight.items()}
teacher.load_state_dict(weight)
del weight
teacher.cuda()
teacher.float()
# load pretrained

# Fix the VGG weights first, and then the weights will be released
for i in range(len(student.module.model0.features)-3):
    for param in student.module.model0.features[i].parameters():
        param.requires_grad = False

trainable_vars = [param for param in student.parameters() if param.requires_grad]
optimizer = torch.optim.SGD(trainable_vars, lr=args.lr,
                           momentum=args.momentum,
                           weight_decay=args.weight_decay,
                           nesterov=args.nesterov)     

for epoch in range(5):
    # train for one epoch
    train_loss = train(train_loader, student, teacher, optimizer, epoch)

    # evaluate on validation set
    val_loss = validate(val_loader, student, teacher, epoch)
    writer.add_scalars('data/scalars', {'val_loss': val_loss, 'train_loss': train_loss}, epoch)
    torch.cuda.empty_cache()
    print(f'Epoch: [{epoch}] train loss: {train_loss}, val loss: {val_loss}')  

# Release all weights                                   
for param in student.module.parameters():
    param.requires_grad = True

trainable_vars = [param for param in student.parameters() if param.requires_grad]
optimizer = torch.optim.SGD(trainable_vars, lr=args.lr,
                           momentum=args.momentum,
                           weight_decay=args.weight_decay,
                           nesterov=args.nesterov)          
                                                    
lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=3, min_lr=0, eps=1e-08)

best_val_loss = np.inf

#########

model_save_filename = 'lib/network/weight/best_pose_Distilation.pth'
for epoch in range(5, args.epochs):

    # train for one epoch
    train_loss = train(train_loader, student, teacher, optimizer, epoch)

    # evaluate on validation set
    val_loss = validate(val_loader, student, teacher, epoch)
    
    lr_scheduler.step(val_loss)   

    writer.add_scalars('data/scalars', {'val_loss': val_loss, 'train_loss': train_loss}, epoch)                    
    
    print(f'Epoch: [{epoch}] train loss: {train_loss}, val loss: {val_loss}')
    
    is_best = val_loss < best_val_loss
    best_val_loss = min(val_loss, best_val_loss)
    if is_best:
        print(f'[TRACE] Update {model_save_filename}')
        torch.save(student.state_dict(), model_save_filename)
          
    torch.cuda.empty_cache()
    
writer.export_scalars_to_json(os.path.join(os.path.abspath(args.model_path), "tensorboard/all_scalars.json"))
writer.close()    