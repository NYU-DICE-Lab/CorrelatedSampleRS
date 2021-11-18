"""
Training script
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10, ImageNet
from torchvision.transforms.transforms import ToTensor
from pathlib import Path


import matplotlib.pyplot as plt
import time
import os
from pathlib import Path
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.models import create_model
from smoothadv.core import Smooth
from smoothadv.attacks import DDN, PGD_L2
from smoothadv.patch_model import PreprocessLayer
from smoothadv.patch_model import PatchModel
from smoothadv.train_utils import AverageMeter, accuracy, requires_grad_

def build_parser():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('dataset', type=str, choices=DATASETS)
    parser.add_argument('-mt', '--mtype', help='Model type',
                        choices=timm.list_models(pretrained=True))
    parser.add_argument('-mpath', '--mpath', help='Path to model', default=None, type=str)
    parser.add_argument('-dpath', help='Data path',
                        default='/data/datasets/Imagenet/val')
    parser.add_argument(
        '-o', '--outdir', help='Output directory', default='results/')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained imagenet model from TIMM')
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--batch', default=256, type=int, metavar='N',
                        help='batchsize (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        help='initial learning rate', dest='lr')
    parser.add_argument('--lr_step_size', type=int, default=30,
                        help='How often to decrease learning by gamma.')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='LR is multiplied by gamma on schedule.')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--noise_sd', default=0.0, type=float,
                        help="standard deviation of Gaussian noise for data augmentation")
    parser.add_argument('--gpu', default=None, type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--use-unlabelled', action='store_true',
                        help='Use unlabelled data via self-training.')
    parser.add_argument('--self-training-weight', type=float, default=1.0,
                        help='Weight of self-training.')
    parser.add_argument('-ps', '--patch_size',
                        help='Patch size', default=224, type=int)
    parser.add_argument('-pstr', '--patch_stride',
                        help='Patch Stride', default=1, type=int)
    parser.add_argument('-np', '--num_patches',
                        help='Maximum number of patches to consider for patch ensemble', type=int, default=10000)
    parser.add_argument(
        '-patch', '--patch', help='use patche-wise ensembling model (default smoothing without patches).',
        action='store_true')
    #####################
    # Attack params
    parser.add_argument('--adv-training', action='store_true')
    parser.add_argument('--attack', default='DDN',
                        type=str, choices=['DDN', 'PGD'])
    parser.add_argument('--epsilon', default=64.0, type=float)
    parser.add_argument('--num-steps', default=10, type=int)
    parser.add_argument('--warmup', default=1, type=int, help="Number of epochs over which \
    -                    the maximum allowed perturbation increases linearly from zero to args.epsilon.")
    parser.add_argument('--num-noise-vec', default=1, type=int,
                        help="number of noise vectors to use for finding adversarial examples. `m_train` in the paper.")
    parser.add_argument('--train-multi-noise', action='store_true',
                        help="if included, the weights of the network are optimized using all the noise samples. \
    -                       Otherwise, only one of the samples is used.")
    parser.add_argument('--no-grad-attack', action='store_true',
                        help="Choice of whether to use gradients during attack or do the cheap trick")

    # PGD-specific
    parser.add_argument('--random-start', default=True, type=bool)

    # DDN-specific
    parser.add_argument('--init-norm-DDN', default=256.0, type=float)
    parser.add_argument('--gamma-DDN', default=0.05, type=float)
    return parser


def build_model(args, smooth=True, patchify=True, pretrained=True):
    base_model = create_model(args.mtype, pretrained=pretrained)
    config = resolve_data_config({}, model=base_model)
    # Hardocoded to ensure additional patches for now.
    config['input_size'] = (3, 256, 256)
    preprocess = PreprocessLayer(config)
    if patchify:
        print('patchify')
        # print('args', args.patch_size, args.patch_stride)
        base_model = PatchModel(base_model, num_patches=args.num_patches,
                                patch_size=args.patch_size, patch_stride=args.patch_stride)
    if smooth:
        model = Smooth(nn.Sequential(preprocess, base_model), num_classes=1000,
                       sigma=args.sigma)  # num classes hardocded for imagenet
    else:
        model = base_model
    return model


def main():

    parser = build_parser()
    args = parser.parse_args()

    args.epsilon = args.epsilon / 255.0
    args.init_norm_DDN = args.init_norm_DDN / 255.0

    # Set the random seed manually for reproducibility.
    torch.manual_seed(0)
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    outdir = Path(args.outdir)
    outfile = open(
        outdir / f'logs_{args.mtype}_{args.sigma}_{args.patch_size}_{args.patch_stride}.log', 'w')
    model_path = args.outdir / f'model_{args.mtype}_{args.sigma}_{args.patch_size}_{args.patch_stride}.pth'
    #print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=outfile, flush=True)

    ################################################################################
    # Load data
    ################################################################################
    if args.dataset == 'imagenet':
        imagenet_train = ImageNet(
            root=args.dpath, split='train', transform=ToTensor())
        train_dl = DataLoader(
            imagenet_train, batch_size=args.batch, shuffle=True, num_workers=args.workers)
        imagenet_val = ImageNet(
            root=args.dpath, split='val', transform=ToTensor())
        val_dl = DataLoader(imagenet_val, batch_size=args.batch,
                            shuffle=False, num_workers=args.workers)
    elif args.dataset == 'cifar10' or args.dataset == 'cifar100':
        raise Exception('Not implemented yet!')
    else:
        raise Exception('Unknown dataset!')

    ################################################################################
    # Build model
    ################################################################################
    model = build_model(args, patchify=args.patch, pretrained=args.pretrained)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.gamma)
    if args.mpath:
        print(f'Loading model from {args.mpath}')
        saved_data = torch.load(args.mpath)
        optimizer.load_state_dict(saved_data['optimizer'])
        model.load_state_dict(saved_data['state_dict'])

    # if args.pretrained_model != '':
    #     assert args.arch == 'cifar_resnet110', 'Unsupported architecture for pretraining'
    #     checkpoint = torch.load(args.pretrained_model)
    #     model = get_architecture(checkpoint["arch"], args.dataset)
    #     model.load_state_dict(checkpoint['state_dict'])
    #     model[1].fc = nn.Linear(64, get_num_classes('cifar10')).cuda()
    # else:
    #     model = get_architecture(args.arch, args.dataset) 
    if args.attack == 'PGD':
        print('Attacker is PGD')
        attacker = PGD_L2(steps=args.num_steps, device='cuda', max_norm=args.epsilon)
    elif args.attack == 'DDN':
        print('Attacker is DDN')
        attacker = DDN(steps=args.num_steps, device='cuda', max_norm=args.epsilon, 
                    init_norm=args.init_norm_DDN, gamma=args.gamma_DDN)
    else:
        raise Exception('Unknown attack')
    
    for epoch in range(args.epochs):
        scheduler.step(epoch)
        attacker.max_norm = np.min([args.epsilon, (epoch + 1) * args.epsilon/args.warmup])
        attacker.init_norm = np.min([args.epsilon, (epoch + 1) * args.epsilon/args.warmup])

        before = time.time()
        train_loss, train_acc = train(args, train_dl, model, criterion, optimizer, epoch, args.noise_sd, attacker)
        test_loss, test_acc, test_acc_normal = test(args, val_dl, model, criterion, args.noise_sd, attacker)
        after = time.time()

        if args.adv_training:
            print(f'{epoch}\t{after - before}\t{scheduler.get_lr()[0]}\t{train_loss}\t{test_loss}\t{train_acc}\t{test_acc}\t{test_acc_normal}', file=outfile, flush=True)
        else:
            print(f'{epoch}\t{after - before}\t{scheduler.get_lr()[0]}\t{train_loss}\t{test_loss}\t{train_acc}\t{test_acc}', file=outfile, flush=True)
        torch.save({
            'epoch': epoch + 1,
            'arch': args.mtype,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, model_path)

def train(args, loader: DataLoader, model: torch.nn.Module, criterion, optimizer: nn.optim.Optimizer, 
        epoch: int, noise_sd: float, attacker: Attacker=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    # switch to train mode
    model.train()
    requires_grad_(model, True)

    for i, (batch, dataLoaderIdx) in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)     

        mini_batches = get_minibatches(batch, args.num_noise_vec)
        noisy_inputs_list = []
        for inputs, targets in mini_batches:
            inputs = inputs.cuda()
            targets = targets.cuda()

            inputs = inputs.repeat((1, args.num_noise_vec, 1, 1)).view(batch[0].shape)

            # augment inputs with noise
            noise = torch.randn_like(inputs, device='cuda') * noise_sd

            if args.adv_training:
                requires_grad_(model, False)
                model.eval()
                inputs = attacker.attack(model, inputs, targets, 
                                        noise=noise, 
                                        num_noise_vectors=args.num_noise_vec, 
                                        no_grad=args.no_grad_attack
                                        )
                model.train()
                requires_grad_(model, True)
            
            if args.train_multi_noise:
                noisy_inputs = inputs + noise

                targets = targets.unsqueeze(1).repeat(1, args.num_noise_vec).reshape(-1,1).squeeze()
                outputs = model(noisy_inputs)
                loss = criterion(outputs, targets)

                acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
                losses.update(loss.item(), noisy_inputs.size(0))
                top1.update(acc1.item(), noisy_inputs.size(0))
                top5.update(acc5.item(), noisy_inputs.size(0))
                    
                # compute gradient and do SGD step
                optimizer.zero_grad()
                # if dataLoaderIdx == PSEUDO_LABELLED:
                #     loss *= args.self_training_weight
                loss.backward()
                optimizer.step()
            
            else:
                inputs = inputs[::args.num_noise_vec] # subsample the samples
                noise = noise[::args.num_noise_vec]
                # noise = torch.randn_like(inputs, device='cuda') * noise_sd
                noisy_inputs_list.append(inputs + noise)

        if not args.train_multi_noise:
            noisy_inputs = torch.cat(noisy_inputs_list)
            targets = batch[1].cuda()
            assert len(targets) == len(noisy_inputs)

            outputs = model(noisy_inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), noisy_inputs.size(0))
            top1.update(acc1.item(), noisy_inputs.size(0))
            top5.update(acc5.item(), noisy_inputs.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            # if dataLoaderIdx == PSEUDO_LABELLED:
            #     loss *= args.self_training_weight
            loss.backward()
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))

    return (losses.avg, top1.avg)


def test(args, loader: DataLoader, model: torch.nn.Module, criterion, noise_sd: float, attacker: Attacker=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    top1_normal = AverageMeter()
    end = time.time()

    # switch to eval mode
    model.eval()
    requires_grad_(model, False)


    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader):
            # measure data loading time
            data_time.update(time.time() - end)

            inputs = inputs.cuda()
            targets = targets.cuda()

            # augment inputs with noise
            noise = torch.randn_like(inputs, device='cuda') * noise_sd
            noisy_inputs = inputs + noise
            
            # compute output
            if args.adv_training:
                normal_outputs = model(noisy_inputs)
                acc1_normal, _ = accuracy(normal_outputs, targets, topk=(1, 5))
                top1_normal.update(acc1_normal.item(), inputs.size(0))

                with torch.enable_grad():
                    inputs = attacker.attack(model, inputs, targets, noise=noise)
                # noise = torch.randn_like(inputs, device='cuda') * noise_sd
                noisy_inputs = inputs + noise

            outputs = model(noisy_inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1.item(), inputs.size(0))
            top5.update(acc5.item(), inputs.size(0))


            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5))

        if args.adv_training:
            return (losses.avg, top1.avg, top1_normal.avg)
        else:
            return (losses.avg, top1.avg, None)