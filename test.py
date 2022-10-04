import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
import cv2
from lib.model import Net
from utils.data import test_dataset
import time

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--gpu_id',   type=str, default='0', help='select gpu id')
parser.add_argument('--test_path', type=str, default='./dataset/test/', help='test dataset path')
# parser.add_argument('--load',    type=str, default='/content/TBINet/ckpt/xx.pth',
#                     help='train from checkpoints')
opt = parser.parse_args()

dataset_path = opt.test_path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# load the model
model = Net()
model.cuda()

ckpt = torch.load(opt.load)
model.load_state_dict(ckpt['state_dict'])
model.eval()

# test
test_datasets = ['NJU2K', 'NLPR']


for dataset in test_datasets:
    print('testing... ', dataset)
    save_path = './test_maps/TBINet/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    image_root = dataset_path + dataset + '/RGB/'
    gt_root = dataset_path + dataset + '/GT/'
    depth_root = dataset_path + dataset + '/depth/'
    test_loader = test_dataset(image_root, gt_root, depth_root, opt.testsize)
    for i in range(test_loader.size):
        image, gt, depth, name, image_for_post = test_loader.load_data()

        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        depth = depth.cuda()
        pre_res = model(image, depth)
        res = pre_res[0]
        res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        cv2.imwrite(save_path+name, res*255)
    print('Test Done!')
