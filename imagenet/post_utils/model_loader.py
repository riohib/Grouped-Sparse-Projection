import argparse
import os
import random
import shutil
import time
import warnings
import numpy as np
import sys

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
from datetime import datetime

import logging

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.autograd import Variable
import util

#os.makedirs('saves', exist_ok=True)
warnings.filterwarnings("ignore")


import sys
sys.path.append("../")
sys.path.append("../..")
import utils_gsp.gpu_projection as gsp_gpu
import utils_gsp.padded_gsp as gsp_model
import utils_gsp.sps_tools as sps_tools


# ===================================================================================
# The Loader has Not yet been adapted to IMAGENET models. So might cause some issues:
# But it works with Resnet models

def load_model(model_path, model_type):
    model_names = sorted(name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name]))

    # Use CUDA
    #os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else 'cpu')

    best_acc = 0  # best test accuracy



    # Data
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    # if args.dataset == 'cifar10':
    dataloader = datasets.CIFAR10
    num_classes = 10
    # else:
    #     dataloader = datasets.CIFAR100
    #     num_classes = 100


    trainset = dataloader(root='../data', train=True, download=True, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers= 4)

    testset = dataloader(root='../data', train=False, download=False, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=100, shuffle=False, num_workers= 4)

    # Model
    print("==> creating model '{}'".format('resnet'))
    model = models.__dict__['resnet'](
                num_classes=num_classes,
                depth=56,
                block_name='BasicBlock',
            )

    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)

    if model_type == 'baseline':
        # Loading Basline Models
        title = 'cifar-10-' + 'resnet'
        # model_path = 'cifar/results/resnet-56_gsp0.99_/2020-11-06_03-05-12/model.pth'
        checkpoint = torch.load(os.path.join('', './checkpoints/cifar10/resnet-56-baseline/model_best'+'.pth.tar'))
        model.load_state_dict(checkpoint['state_dict'])
    
    elif model_type == 'sparse-model':
        ## Loading Sparsified Models
        title = 'cifar-10-' + 'resnet'
        # model_path = './results/resnet-56_global_gsp-all_0.8_/11-16_20-20/model.pth'
        model.load_state_dict(torch.load(model_path))

    return model





# ===================================================================================

class Args:
    data = '/data/users2/rohib/github/imagenet-data'
    arch = 'resnet50'
    reg = 0
    decay = 4e-5
    workers = 16
    epochs = 5
    start_epoch = 0
    batch_size = 256
    lr = 0.001
    lr_decay = 0.1
    lr_int = 30
    momentum = 0.9
    weight_decay = 1e-4
    print_freq = 100
    resume = False
    evaluate = False
    pretrained = ''
    world_size = 1
    dist_url = 'tcp://224.66.41.62:23456'
    dist_backend = 'gloo'
    seed = None
    gpu = None
    sensitivity = 1e-4
    sps = 0.9

# ===================================================================================


global args, best_prec1, save_path
args = Args
best_prec1 = 0

if args.resume:
    if os.path.exists(args.resume):
        save_path = args.resume
        logging.basicConfig(filename=os.path.join(save_path, datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+'log.txt'), level=logging.INFO)
        logger = logging.getLogger('main')
        logger.addHandler(logging.StreamHandler())
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
else:        
    save_path = os.path.join('./results', str(args.arch)+'_'+str(args.decay)+'_'+str(args.reg), datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        raise OSError('Directory {%s} exists. Use a new one.' % save_path)
    logging.basicConfig(filename=os.path.join(save_path, 'log.txt'), level=logging.INFO)
    logger = logging.getLogger('main')
    logger.addHandler(logging.StreamHandler())

logger.info("Saving to %s", save_path)
logger.info("Running arguments: %s", args)


def main(args):

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    if args.gpu is not None:
        model = model.cuda(args.gpu)
    elif args.distributed:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            if args.arch.startswith('alexnetv2'):
                pass
            else:    
                model.features = torch.nn.DataParallel(model.features)
                model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    input_size = 224

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
    #print(model)
    #util.print_model_parameters(model)

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=256, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return


    curves = np.zeros((args.epochs*(len(train_loader)//args.print_freq),7))
    valid = np.zeros((args.epochs,3))
    step = 0

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.exists(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            model.load_state_dict(torch.load(os.path.join(args.resume, 'model.pth')))
            curves_load = np.loadtxt(os.path.join(args.resume, 'curves.dat'))
            valid_load = np.loadtxt(os.path.join(args.resume, 'valid.dat'))
            args.start_epoch = np.count_nonzero(valid_load[:,1])
            step = args.start_epoch*(len(train_loader)//args.print_freq)
            
            curves[0:step,:] = curves_load[0:step,:]
            valid[0:args.start_epoch,:] = valid_load[0:args.start_epoch,:]
            
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, args.start_epoch))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        curves,step = train(train_loader, model, criterion, optimizer, epoch, args.reg, args.decay, curves, step)

        # evaluate on validation set
        valid[epoch, 0] = epoch
        valid[epoch, 1], valid[epoch, 2] = validate(val_loader, model, criterion)
        
        # plot training curve
        np.savetxt(os.path.join(save_path, 'curves.dat'), curves)
        
        clr1 = (0.5, 0., 0.)
        clr2 = (0.0, 0.5, 0.)
        fig, ax1 = plt.subplots()
        fig2, ax3 = plt.subplots()
        ax2 = ax1.twinx()
        ax4 = ax3.twinx()
        ax1.set_xlabel('steps')
        ax1.set_ylabel('Loss', color=clr1)
        ax1.tick_params(axis='y', colors=clr1)
        ax2.set_ylabel('Total loss', color=clr2)
        ax2.tick_params(axis='y', colors=clr2)
        
        ax3.set_xlabel('steps')
        ax3.set_ylabel('Sparsity', color=clr1)
        ax3.tick_params(axis='y', colors=clr1)
        ax4.set_ylabel('Reg', color=clr2)
        #ax4.yscale('log')
        ax4.tick_params(axis='y', colors=clr2)
        
        start = 0
        end = step
        markersize = 12
        coef = 2.
        ax1.plot(curves[start:end, 0], curves[start:end, 1], '--', color=[c*coef for c in clr1], markersize=markersize)
        ax2.plot(curves[start:end, 0], curves[start:end, 6], '-', color=[c*coef for c in clr2], markersize=markersize)
        ax3.plot(curves[start:end, 0], curves[start:end, 3], '--', color=[c*1. for c in clr1], markersize=markersize)
        ax3.plot(curves[start:end, 0], curves[start:end, 4], '-', color=[c*1.5 for c in clr1], markersize=markersize)
        ax3.plot(curves[start:end, 0], curves[start:end, 5], '-', color=[c*2. for c in clr1], markersize=markersize)
        ax4.plot(curves[start:end, 0], curves[start:end, 2], '-', color=[c*coef for c in clr2], markersize=markersize)
        
        #ax2.set_ylim(bottom=20, top=100)
        ax1.legend(('Train loss'), loc='lower right')
        ax2.legend(('Total loss'), loc='lower left')
        fig.savefig(os.path.join(save_path, 'loss-vs-steps.pdf'))
        
        #ax4.set_ylim(bottom=20, top=100)
        ax3.legend(('Elt_sparsity','Filter_sparsity','Average_sparsity'), loc='lower right')
        ax4.legend(('Reg'), loc='lower left')
        fig2.savefig(os.path.join(save_path, 'sparsity-vs-steps.pdf'))
        
        # plot validation curve
        np.savetxt(os.path.join(save_path, 'valid.dat'), valid)
        
        fig3, ax5 = plt.subplots()
        ax6 = ax5.twinx()
        ax5.set_xlabel('epochs')
        ax5.set_ylabel('Acc@1', color=clr1)
        ax5.tick_params(axis='y', colors=clr1)
        ax6.set_ylabel('Acc@5', color=clr2)
        ax6.tick_params(axis='y', colors=clr2)
        
        start = 0
        end = epoch+1
        markersize = 12
        coef = 2.
        ax5.plot(valid[start:end, 0], valid[start:end, 1], '--', color=[c*coef for c in clr1], markersize=markersize)
        ax6.plot(valid[start:end, 0], valid[start:end, 2], '-', color=[c*coef for c in clr2], markersize=markersize)
        
        #ax2.set_ylim(bottom=20, top=100)
        ax5.legend(('Acc@1'), loc='lower right')
        ax6.legend(('Acc@5'), loc='lower left')
        fig3.savefig(os.path.join(save_path, 'accuracy-vs-epochs.pdf'))
        
        torch.save(model.state_dict(), os.path.join(save_path, 'model.pth'))

def train(train_loader, model, criterion, optimizer, epoch, reg_type, decay, curves, step):
    device = torch.device("cuda")
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    total_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()
    itr = 0

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        reg = 0.0
        if decay:
            for name, param in model.named_parameters():
                if param.requires_grad and len(list(param.size()))>1 and 'weight' in name and torch.sum(torch.abs(param))>0:
                    if reg_type==1:    
                        reg += torch.sum(torch.sqrt(torch.sum(param**2,0)))+torch.sum(torch.sqrt(torch.sum(param**2,1)))
                    elif reg_type==2:
                        reg += (torch.sum(torch.sqrt(torch.sum(param**2,0)))+torch.sum(torch.sqrt(torch.sum(param**2,1))))/torch.sqrt(torch.sum(param**2))
                    elif reg_type==3:
                        reg += ( (torch.sum(torch.sqrt(torch.sum(param**2,0)))**2) + (torch.sum(torch.sqrt(torch.sum(param**2,1)))**2) )/torch.sum(param**2)    
                    else:
                        reg = 0.0         
        total_loss = loss+decay*reg

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        total_losses.update(total_loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        device = torch.device("cuda") 

        #=========================================================================
        # sparse-projection
        if (itr % 200 == 0): #gsp every 200 iteration
            print("GSP-Post-ing: itr:  " + str(itr) + 'with sps: ' +str(args.sps))
            sps_tools.gsp_imagenet_partial(model, args.sps, gsp_func = gsp_gpu)
        itr+=1
        #=========================================================================

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i and i % args.print_freq == 0:
            nonzero = total = 0
            filter_count = filter_total = 0
            total_sparsity = total_layer = 0
            for name, p in model.named_parameters():
                if 'weight' in name and len(list(p.size()))>1:
                    tensor = p.data.cpu().numpy()
                    threshold = args.sensitivity
                    new_mask = np.where(abs(tensor) < threshold, 0, tensor)
                    #p.data = torch.from_numpy(new_mask).to(device)
                    tensor = np.abs(new_mask)
                    nz_count = np.count_nonzero(tensor)
                    total_params = np.prod(tensor.shape)
                    nonzero += nz_count
                    total += total_params
                    
                    if len(tensor.shape)==4:
                        dim0 = np.sum(np.sum(tensor, axis=0),axis=(1,2))
                        dim1 = np.sum(np.sum(tensor, axis=1),axis=(1,2))
                        nz_count0 = np.count_nonzero(dim0)
                        nz_count1 = np.count_nonzero(dim1)
                        filter_count += nz_count0*nz_count1
                        filter_total += len(dim0)*len(dim1)
                        total_sparsity += 1-(nz_count0*nz_count1)/(len(dim0)*len(dim1))
                        total_layer += 1
                    if len(tensor.shape)==2:
                        dim0 = np.sum(tensor, axis=0)
                        dim1 = np.sum(tensor, axis=1)
                        nz_count0 = np.count_nonzero(dim0)
                        nz_count1 = np.count_nonzero(dim1)
                        filter_count += nz_count0*nz_count1
                        filter_total += len(dim0)*len(dim1)
                        total_sparsity += 1-(nz_count0*nz_count1)/(len(dim0)*len(dim1))
                        total_layer += 1
                    
            elt_sparsity = (total-nonzero)/total
            input_sparsity = (filter_total-filter_count)/filter_total
            output_sparsity = total_sparsity/total_layer
            
            curves[step, 0] = len(train_loader)*epoch+i
            curves[step, 1] = losses.avg
            curves[step, 2] = reg
            curves[step, 3] = elt_sparsity
            curves[step, 4] = input_sparsity
            curves[step, 5] = output_sparsity
            curves[step, 6] = total_losses.avg
            
            step += 1        
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Total {total.val:.4f} ({total.avg:.4f})\t'
                  'Reg {reg:.4f}\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t Sparsity elt: {elt_sparsity:.3f} str: {input_sparsity:.3f} str_avg: {output_sparsity:.3f}'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, total=total_losses, 
                   reg=reg, top1=top1, top5=top5, elt_sparsity=elt_sparsity, input_sparsity=input_sparsity, output_sparsity=output_sparsity))
                   
    return curves, step




# ========================================================================================
def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    adv_top1 = AverageMeter()
    adv_top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 100 == 0:
                logger.info('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        logger.info(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (args.lr_decay ** (epoch // args.lr_int))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res