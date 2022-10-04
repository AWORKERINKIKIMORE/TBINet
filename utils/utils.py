import torch
import shutil
import torch.nn.functional as F


def structure_loss(pred, mask):
    weit = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask)
    wbce = (weit*wbce).sum(dim=(2, 3))/weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1-(inter+1)/(union-inter+1)

    return (wbce+wiou).mean()


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** ((epoch - 1) // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay*init_lr
        lr = param_group['lr']
    return lr


def adjust_lr_2(optimizer, scheduler, epoch, opt):
    if scheduler == 'step':
        decay = opt.decay_rate ** ((epoch - 1) // opt.decay_epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = decay*opt.lr
            lr = param_group['lr']
    elif scheduler == 'poly':
        decay = (1-(epoch/opt.epoch)) ** opt.power
        for param_group in optimizer.param_groups:
            param_group['lr'] = decay*opt.lr
            lr = param_group['lr']

    return lr


def copyfile(infile, outfile):
    try:
        shutil.copyfile(infile, outfile)
    except:
        print('''Can't open this file''')
        return
