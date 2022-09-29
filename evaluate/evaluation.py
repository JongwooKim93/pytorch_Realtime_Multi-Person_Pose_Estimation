import unittest
import torch
from collections import OrderedDict
from evaluate.coco_eval import run_eval
from lib.network.rtpose import get_model, use_vgg
from lib.network.openpose import OpenPose_Model, use_vgg
from torch import load

#Notice, if you using the
with torch.autograd.no_grad():
    # this path is with respect to the root of the project
    weight_name = './lib/network/weight/best_pose_Distilation.pth'

    weight = {k.replace('module.', ''): v for k, v in torch.load(weight_name).items()}

    model = get_model(trunk='mobilenet')
    #model = openpose = OpenPose_Model(l2_stages=4, l1_stages=2, paf_out_channels=38, heat_out_channels=19)
    #model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(weight)
    model.eval()
    model.float()
    model = model.cuda()

    # The choice of image preprocessing include: 'rtpose', 'inception', 'vgg' and 'ssd'.
    # If you use the converted model from caffe, it is 'rtpose' preprocess, the model trained in
    # this repo used 'vgg' preprocess
    run_eval(image_dir= '/data/coco/images/val2017', anno_file = '/data/coco/annotations/person_keypoints_val2017.json', vis_dir = '/data/coco/images/vis_val2017JW', model=model, preprocess='vgg')


