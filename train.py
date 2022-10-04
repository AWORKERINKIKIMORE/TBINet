import argparse
import os
import torch
from utils.data import get_loader, test_dataset
import numpy as np
import random
import torch.backends.cudnn as cudnn
from lib.model import Net
import torch.nn.functional as F
from utils.utils import clip_gradient, copyfile, structure_loss, adjust_lr
import logging
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def options():
    parser = argparse.ArgumentParser()

    # data path
    parser.add_argument('--rgb_path', type=str, default='./dataset/train/RGB/', help='RGB images path for train')
    parser.add_argument('--depth_path', type=str, default='./dataset/train/depth/', help='Depth images path for train')
    parser.add_argument('--gt_path', type=str, default='./dataset/train/GT/', help='Gt images path for train')
    parser.add_argument('--val_rgb_path', type=str, default='./dataset/test/NJU2K/RGB/',
                        help='rgb images path for test')
    parser.add_argument('--val_depth_path', type=str, default='./dataset/test/NJU2K/depth/',
                        help='depth images path for test')
    parser.add_argument('--val_gt_path', type=str, default='./dataset/test/NJU2K/GT/', help='gt images path for test')
    parser.add_argument('--save_path', type=str, default='./ckpt/', help='save path')
    #parser.add_argument('--drive_path', type=str,
    #                    default='/content/drive/MyDrive/train_files/XX/', help='google drive path')
    # basic setting
    parser.add_argument('--epoch',       type=int,     default=160,   help='epoch number')
    parser.add_argument('--batchsize',   type=int,     default=10,    help='training batch size')
    parser.add_argument('--trainsize',   type=int,     default=352,   help='training dataset size')
    parser.add_argument('--clip',        type=float,   default=0.5,   help='gradient clipping margin')
    # training setting
    parser.add_argument('--lr',          type=float,   default=1e-4,     help='learning rate')
    parser.add_argument('--lr_decay',    type=str,     default='step',   help='lr decay mode')
    parser.add_argument('--decay_rate',  type=float,   default=0.2,      help='gamma for step decay')
    parser.add_argument('--decay_epoch', type=int,     default=60,       help='milestones for step decay')
    parser.add_argument('--load',        type=str,     default=None,     help='train from checkpoints')
    parser.add_argument('--wd',          type=float,   default=1e-4,     help='weight decay')

    opt = parser.parse_args()
    return opt


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = True


def load_data(opt):
    print('==load train==')
    train_rgb_path = opt.rgb_path
    train_gt_path = opt.gt_path
    train_depth_path = opt.depth_path
    train_loader = get_loader(train_rgb_path, train_gt_path, train_depth_path,
                              batchsize=opt.batchsize, trainsize=opt.trainsize)
    val_rgb_path = opt.val_rgb_path
    val_gt_path = opt.val_gt_path
    val_depth_path = opt.val_depth_path
    test_loader = test_dataset(val_rgb_path, val_gt_path, val_depth_path, opt.trainsize)
    return train_loader, test_loader


def setup_logging(opt):
    save_path = opt.save_path
    logging.basicConfig(filename=save_path+'log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]', level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info("Config")
    logging.info('epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};load:{};save_path:{}'.format(
        opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip, opt.load, save_path))


def train(train_loader, model, optimizer, epoch, opt):
    model.train()
    for i, pack in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        images, gts, depths = pack
        images = images.cuda()
        gts = gts.cuda()
        depths = depths.cuda()
        pred = model(images, depths)
        lossF = structure_loss(pred[0], gts)
        lossR = structure_loss(pred[1], gts)
        lossD = structure_loss(pred[2], gts)
        loss = lossF + lossR + lossD
        loss.backward()
        clip_gradient(optimizer, opt.clip)
        optimizer.step()
        if i % 50 == 0 or i == len(train_loader) or i == 1:
            print(' ---Epoch [{:03d}/{:03d}]  Step [{:03d}/{:03d}]  LossF: {:.3f}  LossR: {:.3f}  LossD: {:.3f}  Loss: {:.3f}'
                  .format(epoch, opt.epoch, i, len(train_loader), lossF.data, lossR.data, lossD.data, loss.data))
            logging.info(' ---Epoch [{:03d}/{:03d}]  Step [{:03d}/{:03d}]  LossF: {:.3f}  LossR: {:.3f}  LossD: {:.3f}  Loss: {:.3f}'
                         .format(epoch, opt.epoch, i, len(train_loader), lossF.data, lossR.data, lossD.data, loss.data))
    if (epoch) % 5 == 0:
        torch.save({'optimizer': optimizer.state_dict(),
                    'state_dict': model.state_dict(),
                    'epoch': epoch},
                   opt.save_path + '{}.pth'.format(epoch))


def eval(test_loader, model):
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        for i in range(test_loader.size):
            image, gt, depth, name, img_for_post = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            depth = depth.cuda()
            pre_res = model(image, depth)
            res = pre_res[0]
            res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res-gt))*1.0/(gt.shape[0]*gt.shape[1])

        mae = mae_sum/test_loader.size
        return mae


def main():
    print("==Start training==")
    opt = options()
    setup_seed(2022)
    setup_logging(opt)
    train_loader, test_loader = load_data(opt)

    model = Net()
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)

    if(opt.load is not None):
        ckpt = torch.load(opt.load)
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        init_epoch = ckpt['epoch'] + 1
        print('==load model from ', opt.load, '==')
    else:
        init_epoch = 1
    if not os.path.exists(opt.drive_path):
        os.makedirs(opt.drive_path)
    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'=={total_params:,} total parameters.   ' f'{total_trainable_params:,} training parameters==')

    for epoch in range(init_epoch, opt.epoch+1):
        print(" ")
        print("------------------------------------------------------------------------------------------")
        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        print('lr: {:.2f}e-4'.format(cur_lr * 10000))
        train(train_loader, model, optimizer, epoch, opt)
        mae = eval(test_loader, model)
        print('Epoch: {} MAE: {:.4f}'.format(epoch, mae))
        logging.info('Epoch: {} MAE: {:.4f}'.format(epoch, mae))


if __name__ == '__main__':
    main()
